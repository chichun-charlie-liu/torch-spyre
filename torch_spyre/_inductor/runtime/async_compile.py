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

import json
import tempfile
from typing import Any, Union
import os
import subprocess
from pathlib import Path

from torch._inductor.runtime.runtime_utils import cache_dir
from torch_spyre._C import convert_artifacts
from torch_spyre._inductor.codegen.superdsc import generate_sdsc
from torch_spyre._inductor.constants import SEGMENT_OFFSETS
from . import KernelSpec, ConstantArg, UnimplementedOp
from .kernel_runner import (
    SpyreSDSCKernelRunner,
    SpyreUnimplementedRunner,
)

_argument_names = ["arg0", "arg1", "arg2", "arg3", "arg4", "arg5", "arg6"]


def get_output_dir(kernel_name: str):
    spyre_dir = os.path.join(cache_dir(), "inductor-spyre")
    os.makedirs(spyre_dir, exist_ok=True)
    kernel_output_dir = tempfile.mkdtemp(dir=spyre_dir, prefix=f"{kernel_name}_")
    return kernel_output_dir


class SpyreAsyncCompile:
    def __init__(self) -> None:
        pass

    def sdsc(self, kernel_name: str, ks: Union[KernelSpec | UnimplementedOp]):
        if isinstance(ks, UnimplementedOp):
            print(f"WARNING: Compiling unimplemented {ks.op} to runtime exception")
            return SpyreUnimplementedRunner(kernel_name, ks.op)

        inputs = []
        outputs = []
        arg_mapping = []
        for index, ts in enumerate(ks.args):
            if isinstance(ts, ConstantArg):
                raise RuntimeError("TOOO: implement SDSC generation for constants")
            elif ts.is_input:
                inputs.append(
                    {
                        "name": _argument_names[index],
                        "scale": ks.scales[index],
                        "ddtype": ts.device_layout.device_dtype,
                    }
                )
                arg_mapping.append(ts.arg_index)
            else:
                outputs.append(
                    {
                        "name": _argument_names[index],
                        "scale": ks.scales[index],
                        "ddtype": ts.device_layout.device_dtype,
                    }
                )
                arg_mapping.append(ts.arg_index)
        kernel_descriptor = {
            "name": kernel_name,
            "reduction": ks.is_reduction,
            "op": ks.op,
            "dimensions": ks.dimensions,
            "inputs": inputs,
            "outputs": outputs,
        }
        if ks.op_info is not None:
            kernel_descriptor["op_info"] = ks.op_info
        pointers = dict(zip(_argument_names, SEGMENT_OFFSETS))
        dt_sdsc = generate_sdsc(pointers, **kernel_descriptor)
        kernel_output_dir = get_output_dir(kernel_name)
        subdir = os.path.join(kernel_output_dir, "execute", kernel_name)
        # --- [CL] lx allocation experiments ---
        op_name_sdsc = list(dt_sdsc.keys())[0]
        sch_tree = dt_sdsc[op_name_sdsc]["dscs_"][0][op_name_sdsc]["scheduleTree_"]
        lbl_ds = dt_sdsc[op_name_sdsc]["dscs_"][0][op_name_sdsc]["labeledDs_"]
        comp_op = dt_sdsc[op_name_sdsc]["dscs_"][0][op_name_sdsc]["computeOp_"][0]

        def update_sdsc_mem_alloc_to_lx(ten_idx, lx_offset):
            sch_tree[ten_idx]["name_"] = sch_tree[ten_idx]["name_"].replace("_hbm", "_lx")
            sch_tree[ten_idx]["component_"] = "lx"
            sch_tree[ten_idx]["startAddressCoreCorelet_"]["data_"]["[0, 0, 0]"] = lx_offset
            # TODO need an "LX manager" here to control the address!
            # for now, just offset by inp_size in case inp is loaded onto lx
            if "hbm" in lbl_ds[ten_idx]["memOrg_"]:
                del lbl_ds[ten_idx]["memOrg_"]["hbm"]

        def update_sdsc_to_inplace(out_idx, in_idx, alloc_idx_to_del=None):
            comp_op["outputLabeledDs"][out_idx] = comp_op["inputLabeledDs"][in_idx]
            if alloc_idx_to_del:
                assert (
                    (alloc_idx_to_del+1) == len(sch_tree) == len(lbl_ds)
                ), "Can only delete last allocation, or need to re-number."
                del sch_tree[alloc_idx_to_del]
                del lbl_ds[alloc_idx_to_del]
        
        def load_prev_out3_json_sch_tree(prev_op_idx):
            sch_tree_prev = {}
            if  prev_op_idx is not None:
                prev_ker_name = f"{kernel_name[:-1]}{prev_op_idx}"  # -1 -> what if 2 digits?
                prev_dir = list(Path(kernel_output_dir).parent.glob(f"{prev_ker_name}*"))[0]
                prev_out3 = prev_dir/"execute"/prev_ker_name/"sdsc.out.out.out.json"
                with open(prev_out3, "r") as file:
                    prev_sdsc_o3 = json.load(file)
                    # NOTE hints for lx addr can be found here
                prev_op_name_sdsc = list(prev_sdsc_o3.keys())[0]
                sch_tree_prev = prev_sdsc_o3[prev_op_name_sdsc]["dscs_"][0][prev_op_name_sdsc]["scheduleTree_"]
            return sch_tree_prev
        
        def find_prev_used_lx_addr(prev_o3_sch_tree, partial_name = ""):
            used_lx_addr = {}
            for n in prev_o3_sch_tree:
                nname = n["name_"]
                if not (nname.startswith("transfer") and partial_name in nname):
                    continue

                startAddr = None
                if "dst:lx" in nname:
                    startAddr = n["dstLdsAndLoopOffsets_"][0]["startAddr_"]
                elif "src:lx" in nname:
                    startAddr = n["srcLdsAndLoopOffsets_"]["startAddr_"]

                if isinstance(startAddr, dict):
                    startAddr = startAddr["data_"]["[0, 0, 0]"]
                    used_lx_addr[nname] = int(startAddr)

            return used_lx_addr

        if "softmax" in kernel_name:
            prev_op_map = {"max": None, "sub": 0, "exp": 1, "sum": 2, "realdiv": 3}
            prev_o3_sch_tree = load_prev_out3_json_sch_tree(prev_op_map[ks.op]) 
            prev_used_lx_addr = find_prev_used_lx_addr(prev_o3_sch_tree)
            inp_bytes = ks.dimensions[0] * ks.dimensions[1] * 2

            # handle both (M, N) and (1, N)
            ten0_addr_lx, ten1_addr_lx = None, None
            match ks.op:
                case "max":  # 0:inp MxN, 1: out 1xN
                    ten1_addr_lx = inp_bytes
                case "sub":  # 0:inp MxN, 1: inp 1xN, 2: out MxN
                    # ten0_addr_lx = prev_used_lx_addr['transfer_lds0_src:lxlu_dst:pe']
                    ten1_addr_lx = prev_used_lx_addr["transfer_lds2_src:sfp_dst:lxsu"]
                # case "exp":  # 0:inp MxN, 1: out MxN
                #     ten0_addr_lx = prev_used_lx_addr['transfer_lds2_src:lx_dst:hbm']
                case "sum":  # 0:inp MxN, 1: inp 1xN, 2: out MxN
                    # ten0_addr_lx = prev_used_lx_addr['transfer_lds1_src:lx_dst:hbm']
                    ten1_addr_lx = inp_bytes #  + ten0_addr_lx if ten0_addr_lx else inp_bytes
                case "realdiv":
                    # ten0_addr_lx = prev_used_lx_addr['transfer_lds0_src:lxlu_dst:pe']
                    ten1_addr_lx = prev_used_lx_addr['transfer_lds2_src:sfp_dst:lxsu']

            if ten0_addr_lx is not None:
                update_sdsc_mem_alloc_to_lx(0, ten0_addr_lx)
                # update_sdsc_to_inplace(0, 0, alloc_idx_to_del=1 if ks.op=="exp" else 2)
            if ten1_addr_lx is not None:
                update_sdsc_mem_alloc_to_lx(1, ten1_addr_lx)

            # ONLY reduce HBM access to the small tensor, i.e. max and sum output of size (1, N) 
            # if ks.op in ["max", "sub", "sum", "realdiv"]:
            #     # directly alloc tensor index 1 (output of max and input1 of sub) on lx
            #     update_sdsc_mem_alloc_to_lx(1, inp_bytes)



        os.makedirs(subdir, exist_ok=True)
        with open(os.path.join(subdir, "sdsc.json"), "w") as file:
            print(f"Generating {file.name}")
            json.dump(dt_sdsc, file, indent=2)
        subprocess.run(["dxp_standalone", "-d", kernel_output_dir], check=True)
        convert_artifacts(kernel_output_dir)
        return SpyreSDSCKernelRunner(kernel_name, kernel_output_dir, arg_mapping)

    def wait(self, scope: dict[str, Any]) -> None:
        pass
