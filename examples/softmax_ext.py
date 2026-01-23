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

import torch

DEVICE = torch.device("spyre")
torch.manual_seed(0xAFFE)

# Create input tensor
x = torch.rand(512, 1024, dtype=torch.float16)

# Compute softmax on the cpu
cpu_result = torch.softmax(x, dim=0)

# Send input tensor to device
x_device = x.to(DEVICE)
# buf_device = torch.empty((1,x.shape[1]), dtype=torch.float16).to(DEVICE)

# Compute softmax on the device in eager mode and get the result back to the host
eager_result = torch.softmax(x_device, dim=0).cpu()

compiled_sm = torch.compile(lambda a: torch.softmax(a, dim=0))
compiled_result = compiled_sm(x_device).cpu()


# --- [CL] wrap softmax as a new op -> inductor will not decompose it, so that we may
#           implement a new fused sdsc.json template in torch-spyre
@torch.library.custom_op("spyre::custom_softmax", mutates_args=())
def custom_softmax(input: torch.Tensor, dim: int =0) -> torch.Tensor:
    # add a dummy buf: torch.Tensor?
    return  # torch.softmax(input, dim)

@custom_softmax.register_fake
def _(input, dim=0):
    return torch.empty_like(input)

# compiled_sm = torch.compile(custom_softmax)
# compiled_result = compiled_sm(x_device, dim=0).cpu()


# Print the results and compare them
print(f"CPU result\n{cpu_result}")
print(f"Spyre Eager result\n{eager_result}")
print(f"Spyre Compiled result\n{compiled_result}")
device_delta = torch.abs(eager_result - compiled_result).max()
cpu_delta = torch.abs(compiled_result - cpu_result).max()

print(f"Max delta Compiled Spyre vs. CPU: {cpu_delta}")
print(f"Max delta Compiled Spyre vs. Eager Spyre: {device_delta}")
