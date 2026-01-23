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

# this import will start the runtime
import torch

# x_cpu = torch.tensor([0.12345], dtype=torch.float16)
x_cpu = torch.randn((2, 3), dtype=torch.float16)
x_aiu = x_cpu.to("spyre")
x_aiu2cpu = x_aiu.to("cpu")

print(f"x_cpu value={x_cpu}")  # , device={x_cpu.device}")
print(f"x_aiu value={x_aiu}")  #, device={x_aiu.device}")
print(
    f"\n||x_cpu - x_aiu2cpu|| = {torch.norm(x_cpu-x_aiu2cpu)}\n"
    "NOTE: There will be some numerical errors when moving tensor from CPU(FP16) -> AIU(DL16) -> CPU(FP16).\n"
    "If we do the same thing for CPU->GPU->CPU, norm will be exactly 0!!\n"
)

