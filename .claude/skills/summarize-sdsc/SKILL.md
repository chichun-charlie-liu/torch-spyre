---
name: summarize-sdsc
description: "Analyze and summarize all sdsc.json files from Inductor debug artifacts."
exec: "python3 batch_summarize_sdsc.py"
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
/summarize-sdsc [optional_directory_path]
```

**Default behavior (no arguments):** Automatically analyzes the most recent `/tmp/torchinductor_<username>/` directory where PyTorch Inductor writes debug artifacts. Returns a complete summary without prompts.

**Custom directory:** Optionally provide a directory path to analyze a specific location containing `sdsc_*` subdirectories:
```
/summarize-sdsc /path/to/custom/directory
```

**File discovery:** Automatically searches for all `sdsc_*.json` files (excluding `.out.json` variants) within the directory and its subdirectories.

## Output

Generates a comprehensive summary including:

1. **Operations Summary** — Lists each operation with tensor allocations and components
2. **Tensor Summary Table** — One row per tensor allocation showing layout, stick dimensions, component, memory addresses, and source file
3. **Processing Statistics** — File counts, operation types, and component usage metrics

Stick dimensions are marked with `*` in the layout column for easy identification.

## Examples

```bash
# Analyze the most recent torchinductor directory (default)
/summarize-sdsc

# Specify a custom directory
/summarize-sdsc ./torch-compile-debug

# Use an explicit path
/summarize-sdsc /tmp/torchinductor_cliu/
```
