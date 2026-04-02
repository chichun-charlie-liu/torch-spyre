---
name: summarize-sdsc
description: "Analyze and summarize all sdsc.json files from Inductor debug artifacts."
---

# Summarize SDSC

Analyzes all `sdsc.json` files in a directory and generates a comprehensive summary of:

- Operations and their resource allocation (cores, corelets)
- Tensor allocations with roles (INPUT, OUTPUT, KERNEL)
- Memory component usage and addresses
- Layout and stick dimension information
- Per-operation breakdown with detailed component trees

## Usage

```
/summarize-sdsc [directory_path]
```

Pass the directory containing Inductor debug artifacts (the one with `sdsc_*` subdirectories).

**Default path:** If no directory is provided, defaults to `/tmp/torchinductor_<username>/`, which is where PyTorch Inductor writes debug artifacts.

**File matching:** Automatically searches for all `sdsc*.json` files within the specified directory and its subdirectories.

## Output

The tool generates three main sections:

1. **All Operations Table** — Shows each operation with its tensor allocations, components, and memory addresses
2. **Tensor Summary Table** — One row per tensor allocation with layout and stick information
3. **Processing Statistics** — File counts, operation types, component usage, and resource allocation metrics

## Examples

```bash
# Use default path (/tmp/torchinductor_<username>/)
/summarize-sdsc

# Specify a custom directory
/summarize-sdsc ./torch-compile-debug

# Use an explicit path
/summarize-sdsc /tmp/torchinductor_cliu/
```
