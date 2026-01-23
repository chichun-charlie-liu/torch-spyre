# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import inspect
import sympy
from torch._inductor.codecache import code_hash
from torch._inductor.codegen.simd_kernel_features import SIMDKernelFeatures
from torch._inductor.codegen.wrapper import (
    BufferLike,
    PythonWrapperCodegen,
    SubgraphPythonWrapperCodegen,
)
from torch._inductor.ir import GraphPartitionSignature, ComputedBuffer, Pointwise
from torch._inductor.virtualized import V
from torch._inductor.scheduler import Scheduler, SchedulerNode
from torch._inductor.sizevars import SizeVarAllocator
from .spyre_kernel import SpyreKernel
from .stickify import FixedTiledLayout


class SpyrePythonWrapperCodegen(PythonWrapperCodegen):
    def __init__(self):
        super().__init__()
        V.graph.sizevars._simplify_loops_impl = noop_simplify_loops_impl.__get__(
            V.graph.sizevars, SizeVarAllocator
        )

    @staticmethod
    def create(
        is_subgraph: bool,
        subgraph_name: Optional[str],
        parent_wrapper: Optional[PythonWrapperCodegen],
        partition_signatures: Optional[GraphPartitionSignature] = None,
    ):
        if is_subgraph:
            assert subgraph_name is not None
            assert parent_wrapper is not None
            return SubgraphPythonWrapperCodegen(
                subgraph_name, parent_wrapper, partition_signatures
            )
        return SpyrePythonWrapperCodegen()

    def write_header(self) -> None:
        super().write_header()
        self.imports.splice(
            """
                from torch_spyre._inductor.runtime import ConstantArg, TensorArg, KernelSpec, UnimplementedOp
                from torch_spyre._inductor.runtime.async_compile import SpyreAsyncCompile
                from torch_spyre._C import DataFormats, SpyreTensorLayout, StickFormat, spyre_empty_with_layout
                import subprocess
            """,
            strip=True,
        )
        self.header.writeline("del async_compile")
        self.header.writeline("async_compile = SpyreAsyncCompile()")

    def make_buffer_allocation(self, buffer: BufferLike):
        layout = buffer.get_layout()
        if not isinstance(layout, FixedTiledLayout):
            return super().make_buffer_allocation(buffer)

        name = buffer.get_name()
        codegen_shape_tuple = self.codegen_python_shape_tuple(tuple(layout.size))
        codegen_allocation_shape_tuple = self.codegen_python_shape_tuple(
            tuple(layout.get_allocation_size())
        )
        codegen_stride_tuple = self.codegen_python_shape_tuple(tuple(layout.stride))

        out = (
            f"{name} = spyre_empty_with_layout("
            f"{codegen_allocation_shape_tuple}, "
            f"{codegen_stride_tuple}, "
            f"{layout.dtype}, "
            f"{layout.device_layout!r})"
        )
        if codegen_shape_tuple != codegen_allocation_shape_tuple:
            out = out + f".as_strided({codegen_shape_tuple}, {codegen_stride_tuple})"

        return out

    def generate_fallback_kernel(self, fallback_kernel):
        """
        [CL] Implemention of Spyre custom OP, usually will correspond to a hand-crafted
        superDSC json template.
            1. create a SpyreKernel based on fallback_kernel
            2. follow the flow in <SIMDScheduling>.codegen_node_schedule()
        NOTE assuming our custom Spyre OPs are all registered under "spyre" namespace
        """
        kernel = fallback_kernel.op_overload
        if kernel.namespace != "spyre":
            super().generate_fallback_kernel(fallback_kernel)
            return

        # use `inspect` to retrieve access to some objects not directly available here
        prev_frame = inspect.currentframe().f_back.f_back
        scheduler = prev_frame.f_locals["self"]
        spyre_backend = scheduler.get_backend(scheduler.current_device)

        assert isinstance(scheduler, Scheduler), (
            "check code base and make sure the relative position of callstack frames"
            " remain unchanged. 'self' in two frames before should be Scheduler."
        )

        kwargs = {
            "device": fallback_kernel.layout.device,
            "dtype": fallback_kernel.layout.dtype,
            "inner_fn": fallback_kernel.inputs[0].make_loader(),  # this is "var loader"
            "ranges": fallback_kernel.layout.size,
        }
        match kernel._opname:
            # fallback_kernel does not have .data (ir.Pointwise, ir.Reduction, ...)
            # needed for ComputedBuffer. make one based on kernel's need in match/case
            case "custom_softmax":
                # implement custom fused softmax here
                assert len(fallback_kernel.inputs[0].shape) == 2, "softmax can only handle 2D tensors."
                softmax_dim = fallback_kernel.get_kwargs_value("dim")
                numel, rnumel = fallback_kernel.get_numel(), 1

                # NOTE update kwargs if needed
            case _:
                # default, no reduction
                numel, rnumel = fallback_kernel.get_numel(), 1

        cb = ComputedBuffer(
            name=fallback_kernel.name,
            layout=fallback_kernel.layout,
            data=Pointwise(**kwargs),
        )
        # copy needed (but missing) attr from fallback_kernel to ComputedBuffer
        for attr in ["operation_name", "origin_node", "origins"]:
            setattr(cb, attr, getattr(fallback_kernel, attr))

        # TODO patch sol'n for now, double check how does torch handle softmax "dim"
        cb.softmax_dim = softmax_dim

        dummy_sch_node = SchedulerNode(scheduler, cb)
        node_schedule = [dummy_sch_node]

        # NOTE we could "almost" just call the original func here, i.e.,
        #    spyre_backend.codegen_node_schedule(SIMDKernelFeatures(node_schedule, numel, rnumel))
        # BUT, parsing opname into kernel.spyre_op still requires some works, Hence, use
        # simplified/modified version as a temp patch for now.
        simplified_codegen_node_schedule(
            SIMDKernelFeatures(node_schedule, numel, rnumel),
            spyre_backend,
            fallback_kernel,
        )




def simplified_codegen_node_schedule(kernel_feature, spyre_backend, fallback_kernel):
    """
    This is a modified version of spyre_backend.codegen_node_schedule(). Mainly because
    spyre_backend.codegen_node_schedule_with_kernel() cannot perfectly parse our derived
    SchedulerNode (i.e., node_schedule[0]) and cause some missing attr in spyre_kernel.
    Before we find a cleaner way, manually add those attr back.

    NOTE V.graph.wrapper_code will hold the genereated wrapper_code
    """
    # use default tiling here, backend compiler will handle the details.
    opname = fallback_kernel.op_overload._opname
    numel = kernel_feature.numel
    rnumel = kernel_feature.reduction_numel
    node_schedule = kernel_feature.node_schedule
    spyre_kernel = SpyreKernel(
        spyre_backend.create_tiling([numel], [rnumel]),
        features=kernel_feature,
        tiling_scores=None,
    )

    # This func call will fill in key attrs for spyre_kernel, e.g., .spyre_op,
    # .compute_input, .compute_output, ...
    spyre_backend.codegen_node_schedule_with_kernel(node_schedule, spyre_kernel)

    # --- Main reason to use this patch func, add missing attr back to spyre_kernel ---
    spyre_kernel.compute_op = opname
    spyre_kernel.spyre_op = opname
    spyre_kernel.kernel_name = opname
    spyre_kernel.compute_op_is_reduction = False

    # test to add a dummy buf in compute_input to match with sdsc.json alloc
    # arg1 = fallback_kernel.inputs[1]
    # index = spyre_kernel.compute_inputs[0].index  # same stride -> same index
    # spyre_kernel.compute_inputs.append(TensorAccess(arg1.name, index, arg1.layout))
    # ---------------------------------------------------------------------------------

    with V.set_kernel_handler(spyre_kernel):
        src_code = spyre_kernel.codegen_kernel()
    kernel_name = spyre_backend.define_kernel(src_code, node_schedule, spyre_kernel)
    spyre_kernel.kernel_name = kernel_name
    spyre_kernel.code_hash = code_hash(src_code)

    with V.set_kernel_handler(spyre_kernel):
        node_schedule[0].mark_run()

    spyre_backend.codegen_comment(node_schedule)

    spyre_kernel.call_kernel(kernel_name)
    V.graph.removed_buffers |= spyre_kernel.removed_buffers
    V.graph.inplaced_to_remove |= spyre_kernel.inplaced_to_remove

    spyre_backend.free_buffers_in_scheduler()


def noop_simplify_loops_impl(
    self, index_vars: list[sympy.Symbol], sizes, index_formulas
):
    """
    This is a noop implementation of SizeVarAllocator._simplify_loops_impl.

    We do this because the memory layout of tensors on the Spyre device is not
    entirely visible to Inductor.  Therefore Inductor's understanding of which
    tensor dimensions are actually contiguous is not accurate.
    """
    return sizes, lambda x: x, lambda x: x
