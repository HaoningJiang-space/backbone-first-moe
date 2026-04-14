# FineMoE Runtime Patches

Real-hardware evaluation requires the FineMoE runtime with backbone-first support.
These patches add native resident pinning to the FineMoE C++ engine.

## What the patch changes

| File | Change |
|---|---|
| `core/model/model_topology.h` | Add `bool is_resident` flag to Node |
| `core/prefetch/task_scheduler.cpp` | Skip `is_resident` nodes in eviction |
| `core/prefetch/archer_prefetch_handle.cpp` | Add `PinResidentNodes()` method |
| `core/prefetch/archer_prefetch_handle.h` | Declare `PinResidentNodes()` |
| `core/python/py_archer_prefetch.cpp` | Expose `pin_resident_nodes` to Python |
| `finemoe/runtime/model_offload.py` | `pin_resident_experts()` using native API |
| `finemoe/memory/expert_prefetcher.py` | Skip resident experts in prefetch |
| `finemoe/utils/config.py` | Add `resident_expert_ids_file` config field |
| `finemoe/models/modeling_qwen/modeling_qwen2_moe.py` | Prefetch distance guard |

## How to apply

```bash
# 1. Clone FineMoE
git clone https://github.com/IntelliSys-Lab/FineMoE-EuroSys26.git
cd FineMoE-EuroSys26

# 2. Apply patch
git apply /path/to/backbone-first-moe/patches/finemoe_backbone.patch

# 3. Install (editable mode)
pip install -e .

# 4. Clear JIT cache to trigger C++ rebuild
rm -rf ~/.cache/torch_extensions/*/prefetch

# 5. Verify
python -c "from finemoe.runtime.model_offload import OffloadEngine; print('OK')"
```

The C++ extension will be JIT-compiled on first import after applying the patch.
