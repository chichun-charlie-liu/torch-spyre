#!/usr/bin/env python3
# Copyright 2024 IBM. All Rights Reserved.
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

"""Batch summarize all sdsc.json files from Inductor debug artifacts."""

import json
import sys
from pathlib import Path
from collections import defaultdict


def _extract_tensor_idx_to_role(op_data: dict) -> dict:
  """Extract tensor ldsIdx to role mapping from computeOp_ and primaryDsInfo_.

  Returns a dict mapping ldsIdx to role (INPUT/OUTPUT/KERNEL).
  """
  lds_idx_to_role = {}

  if "computeOp_" in op_data:
    compute_ops = op_data["computeOp_"]
    if isinstance(compute_ops, list):
      compute_op = compute_ops[0] if compute_ops else {}
    else:
      compute_op = compute_ops

    # Extract input tensor indices
    input_indices = set()
    for labeled_ds in compute_op.get("inputLabeledDs", []):
      if "-idx" in labeled_ds:
        idx_str = labeled_ds.split("-idx")[1]
        input_indices.add(int(idx_str))

    # Extract output tensor indices
    output_indices = set()
    for labeled_ds in compute_op.get("outputLabeledDs", []):
      if "-idx" in labeled_ds:
        idx_str = labeled_ds.split("-idx")[1]
        output_indices.add(int(idx_str))

    # Map ldsIdx to role based on usage
    for lds_idx in range(10):  # Reasonable upper limit
      if lds_idx in output_indices:
        lds_idx_to_role[lds_idx] = "OUTPUT"
      elif lds_idx in input_indices:
        lds_idx_to_role[lds_idx] = "INPUT"
      else:
        # Tensor used but not explicitly marked
        if lds_idx in input_indices or lds_idx in output_indices:
          lds_idx_to_role[lds_idx] = "KERNEL"

  # Fall back to primaryDsInfo roles if available and not already mapped
  if "primaryDsInfo_" in op_data:
    for role_idx, (role, _) in enumerate(op_data["primaryDsInfo_"].items()):
      if role_idx not in lds_idx_to_role:
        lds_idx_to_role[role_idx] = role

  return lds_idx_to_role


def _extract_operation_data(op_data: dict) -> dict:
  """Extract common operation data from op_data dict."""
  # Extract components and their details
  components = []
  component_details = []
  if "scheduleTree_" in op_data:
    for node in op_data["scheduleTree_"]:
      if isinstance(node, dict):
        comp = node.get("component_", "")
        if comp:
          components.append(comp)
          node_name = node.get("name_", "")
          node_type = node.get("nodeType_", "")
          component_details.append({
              "name": node_name,
              "type": node_type,
              "component": comp,
          })

  # Extract allocate nodes indexed by ldsIdx
  allocate_nodes = {}
  if "scheduleTree_" in op_data:
    for node in op_data["scheduleTree_"]:
      if isinstance(node, dict) and node.get("nodeType_") == "allocate":
        lds_idx = node.get("ldsIdx_", -1)
        allocate_nodes[lds_idx] = {
            "name": node.get("name_", "").lower().replace("-", "_"),
            "component": node.get("component_", ""),
            "address": "",
        }
        if "startAddressCoreCorelet_" in node:
          addr_data = node["startAddressCoreCorelet_"].get("data_", {})
          if addr_data:
            allocate_nodes[lds_idx]["address"] = str(list(addr_data.values())[0])

  # Get tensor index to role mapping from computeOp_
  lds_idx_to_role = _extract_tensor_idx_to_role(op_data)

  # Extract tensors - one entry per ldsIdx (each allocation)
  tensors = []

  for lds_idx, alloc_node in sorted(allocate_nodes.items()):
    # Get the role for this ldsIdx
    role = lds_idx_to_role.get(lds_idx, "UNKNOWN")

    # Get layout and stick info from primaryDsInfo_ if available
    layout_dims = ""
    stick_dims = ""
    if "primaryDsInfo_" in op_data and role in op_data["primaryDsInfo_"]:
      tensor_data = op_data["primaryDsInfo_"][role]
      layout_dims = ", ".join(tensor_data.get("layoutDimOrder_", []))
      stick_dims = ", ".join(tensor_data.get("stickDimOrder_", []))

    # One allocation per ldsIdx
    tensors.append({
        "name": alloc_node["name"],
        "role": role,
        "layout": layout_dims,
        "sticks": stick_dims,
        "component": alloc_node["component"],
        "address": alloc_node["address"],
    })

  # Extract address
  addr = ""
  if "scheduleTree_" in op_data and op_data["scheduleTree_"]:
    node = op_data["scheduleTree_"][0]
    if "startAddressCoreCorelet_" in node:
      addr_data = node["startAddressCoreCorelet_"].get("data_", {})
      if addr_data:
        addr = str(list(addr_data.values())[0])

  # Extract dimensions
  dims = ""
  if "N_" in op_data:
    n_info = op_data["N_"]
    dim_parts = []
    for k in ["mb_", "out_", "in_", "k_", "h_", "w_"]:
      if k in n_info:
        dim_parts.append(f"{k[:-1]}:{n_info[k]}")
    dims = ", ".join(dim_parts)

  return {
      "components": components,
      "components_str": ", ".join(components),
      "component_details": component_details,
      "tensors": tensors,
      "tensors_str": ", ".join([t["role"] for t in tensors]),
      "address": addr,
      "dims": dims,
  }


def extract_ops_from_sdsc(data: dict, file_name: str) -> list:
  """Extract operations from sdsc.json files (both standard and alternative formats)."""
  operations = []

  # Format 1: Standard format with dscs_ or identity/dscs_
  dscs_list = data.get("dscs_", data.get("identity", {}).get("dscs_", []))

  if dscs_list:
    # Process standard format
    for dsc_idx, dsc_dict in enumerate(dscs_list):
      for op_name, op_data in dsc_dict.items():
        if not isinstance(op_data, dict):
          continue

        op_info = _extract_operation_data(op_data)
        operations.append({
            "file": file_name,
            "op_name": op_name,
            "dsc_idx": dsc_idx,
            "cores": op_data.get("numCoresUsed_", 0),
            "corelets": op_data.get("numCoreletsUsed_", 0),
            **op_info,
        })
  else:
    # Format 2: Alternative format with operation key at top level
    # (e.g., maxnonstick, sub, exp, sumnonstick, realdiv)
    alternative_op_keys = ["maxnonstick", "sub", "exp", "sumnonstick", "realdiv"]
    for op_key in alternative_op_keys:
      if op_key in data:
        op_container = data[op_key]
        if isinstance(op_container, dict) and "dscs_" in op_container:
          nested_dscs = op_container["dscs_"]
          for dsc_idx, dsc_dict in enumerate(nested_dscs):
            for nested_op_name, nested_op_data in dsc_dict.items():
              if not isinstance(nested_op_data, dict):
                continue

              op_info = _extract_operation_data(nested_op_data)
              operations.append({
                  "file": file_name,
                  "op_name": nested_op_name,
                  "dsc_idx": dsc_idx,
                  "cores": nested_op_data.get("numCoresUsed_", 0),
                  "corelets": nested_op_data.get("numCoreletsUsed_", 0),
                  **op_info,
              })

  return operations


def batch_summarize_directory(base_dir_str: str) -> None:
  """Summarize all sdsc.json files in a directory."""
  base_dir = Path(base_dir_str)

  if not base_dir.exists():
    print(f"Error: Directory not found: {base_dir_str}", file=sys.stderr)
    sys.exit(1)

  # Find only sdsc.json files
  sdsc_files = sorted(base_dir.glob("**/sdsc.json"))

  if not sdsc_files:
    print(f"No sdsc.json files found in {base_dir_str}")
    return

  print(f"\n{'='*200}")
  print(f"SDSC Operations Summary - Batch Report")
  print(f"Directory: {base_dir}")
  print(f"Total sdsc.json files found: {len(sdsc_files)}")
  print(f"{'='*200}\n")

  # Collect all operations data
  all_ops = []
  stats = {
      "total_files": len(sdsc_files),
      "files_with_ops": 0,
      "files_skipped": 0,
      "operation_types": defaultdict(int),
      "component_types": defaultdict(int),
  }

  for file_path in sdsc_files:
    try:
      with open(file_path) as f:
        data = json.load(f)
    except (json.JSONDecodeError, IOError):
      stats["files_skipped"] += 1
      continue

    file_rel_path = file_path.relative_to(base_dir)
    file_name = str(file_rel_path)

    ops = extract_ops_from_sdsc(data, file_name)

    if ops:
      stats["files_with_ops"] += 1
      for op in ops:
        stats["operation_types"][op["op_name"]] += 1
        for comp in op["components"]:
          stats["component_types"][comp] += 1
      all_ops.extend(ops)
    else:
      stats["files_skipped"] += 1

  # Print operations table
  if all_ops:
    print("All Operations Table (with Tensor Allocations):")
    print("=" * 300)

    # Use shorter file paths - just the kernel directory
    file_col_width = 35

    header = (
        f"{'Kernel':<{file_col_width}} | {'Op':<15} | {'Cores':<5} | "
        f"{'Tensor':<20} | {'Role':<8} | {'Component':<12} | {'Address':<15} | {'Layout':<20}"
    )
    print(header)
    print("-" * 300)

    for op in all_ops:
      # Extract just the kernel directory name (e.g., "sdsc_fused_0_ahpi_5jb")
      parts = op['file'].split('/')
      kernel_name = parts[0] if parts else op['file']
      file_short = kernel_name if len(kernel_name) <= file_col_width else kernel_name[:file_col_width-3] + "..."

      # If no tensors, show operation summary
      if not op["tensors"]:
        print(
            f"{file_short:<{file_col_width}} | {op['op_name']:<15} | "
            f"{op['cores']:<5} | {'':<20} | {'':<8} | {'':<12} | {op['address']:<15} | {'':<20}"
        )
      else:
        # For each tensor with its allocations, create a row per allocation
        for tensor_idx, tensor in enumerate(op["tensors"]):
          # Show op/file/cores only on first row of operation
          op_label = op['op_name'] if tensor_idx == 0 else ""
          file_label = file_short if tensor_idx == 0 else ""
          cores_label = str(op['cores']) if tensor_idx == 0 else ""

          print(
              f"{file_label:<{file_col_width}} | {op_label:<15} | "
              f"{cores_label:<5} | {tensor['name']:<20} | {tensor['role']:<8} | "
              f"{tensor['component']:<12} | {tensor['address']:<15} | {tensor['layout']:<20}"
          )

    print("=" * 300)

    # Print tensor summary table - one row per unique tensor allocation
    print("\n\nTensor Summary Table (One row per tensor allocation):")
    table_width = 155
    print("=" * table_width)

    tensor_header = (
        f"{'Op':<20} | {'Tensor Name':<20} | {'Role':<10} | {'Layout':<20} | "
        f"{'Sticks':<15} | {'Component':<12} | {'Address':<15}"
    )
    print(tensor_header)
    print("-" * table_width)

    for op_idx, op in enumerate(all_ops):
      op_name = op['op_name']
      if op["tensors"]:
        # Print one row per tensor allocation
        for alloc_idx, tensor in enumerate(op["tensors"]):
          op_label = op_name if alloc_idx == 0 else ""
          print(
              f"{op_label:<20} | {tensor['name']:<20} | {tensor['role']:<10} | "
              f"{tensor['layout']:<20} | {tensor['sticks']:<15} | "
              f"{tensor['component']:<12} | {tensor['address']:<15}"
          )

        # Add separator line between operations (but not after the last one)
        if op_idx < len(all_ops) - 1:
          print("-" * table_width)

    print("=" * table_width)

  # Print statistics
  print("\nProcessing Statistics:")
  print("-" * 80)
  print(f"Total sdsc.json files:           {stats['total_files']}")
  print(f"Files with operations:           {stats['files_with_ops']}")
  print(f"Files without operations:        {stats['files_skipped']}")

  if all_ops:
    print(f"\nOperations Summary:")
    print(f"  Total operations found:        {len(all_ops)}")
    print(f"  Unique operation types:        {len(stats['operation_types'])}")
    for op_type, count in sorted(stats["operation_types"].items()):
      print(f"    • {op_type:<40}: {count:3d}")

    if stats["component_types"]:
      print(f"\nMemory Components:")
      for comp, count in sorted(stats["component_types"].items()):
        print(f"  {comp:<35}: {count:3d}")

    print(f"\nResource Allocation:")
    total_cores = sum(op["cores"] for op in all_ops)
    max_cores = max(op["cores"] for op in all_ops) if all_ops else 0
    avg_cores = total_cores / len(all_ops) if all_ops else 0
    print(f"  Total cores allocated:        {total_cores}")
    print(f"  Max cores per operation:      {max_cores}")
    print(f"  Avg cores per operation:      {avg_cores:.2f}")

    # Detailed operations breakdown
    print(f"\nDetailed Operations Breakdown:")
    print("-" * 80)
    for op in all_ops:
      print(f"\n{op['file']} : {op['op_name']} (DSC {op['dsc_idx']})")
      print(f"  Resources: {op['cores']} cores, {op['corelets']} corelets")
      if op["component_details"]:
        print(f"  Components:")
        for comp in op["component_details"]:
          print(
              f"    • {comp['name']} ({comp['type']}, {comp['component']})"
          )
      if op["tensors"]:
        print(f"  Tensors:")
        for tensor in op["tensors"]:
          print(
              f"    • {tensor['name']} ({tensor['role']}): layout=[{tensor['layout']}], "
              f"sticks=[{tensor['sticks']}] @ {tensor['component']} {tensor['address']}"
          )
      if op["dims"]:
        print(f"  Dimensions: {op['dims']}")
  else:
    print("No operations found in any sdsc.json files.")

  print()


if __name__ == "__main__":
  if len(sys.argv) < 2:
    print("Usage: batch_summarize_sdsc.py <directory_path>", file=sys.stderr)
    sys.exit(1)

  batch_summarize_directory(sys.argv[1])
