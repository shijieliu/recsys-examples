# One host wait per DynamicEmb lookup

This PoC compares Python prefetch implementations on real DynamicEmb training steps. The benchmark temporarily selects a prefetch function on the counted Python runtime from [PR #545](https://github.com/NVIDIA/recsys-examples/pull/545). The module's forward, backward, storage, optimizer, and CUDA operators run normally. The budget is per collection lookup. A step with multiple collections can spend one wait on each.

| Path | Host decision | GPU work |
| --- | --- | --- |
| HBM `baseline` | Keep the miss count on device | #545 runs the counted miss path even when all keys hit. |
| HBM `fused` | Keep the miss count on device | One Triton kernel writes the slot and value-row maps for inserted keys. |
| HBM `one_wait` | Read the miss count after pinning hits | An all-hit batch skips initialization and insertion. A mixed batch uses the fused scatter. |
| Cache `baseline` | Keep counts on device | #545 runs its counted prefetch. This timing control predates the cache refusal check in #557. |
| Cache `early` | Read status just after reserving the missing slots | An all-hit batch also skips loading or initializing backing rows. |
| Cache `late` | Prepare backing rows, reserve slots, then read status | Skip victim movement when there are no evictions. |

Both cache candidates update backing scores for all unique keys, pin hits before reserving slots, and reuse the same list of new keys for backing insertion and initialization. One Triton kernel collects the number of installs, evictions and refusals into a single read. A refused batch completes backing insertion, releases its hit pins and raises before cache writeback. Successful batches capture victims before overwriting their slots.

## Run on one GPU

Start at the root of the PoC checkout in a CUDA development environment with PyTorch, Triton, FBGEMM, TorchRec and pytest. Use a disposable environment for the pinned runtime build since its setup script manages the installed DynamicEmb package. The following build targets H100 (`DEMB_ARCHS=90`).

```bash
POC="$PWD/corelib/dynamicemb/benchmark/one_wait_lookup"
git fetch https://github.com/shijieliu/recsys-examples.git 7141addba327941f7ea802f99938ad72c3970ace
git worktree add --detach ../one-wait-runtime 7141addba327941f7ea802f99938ad72c3970ace
cd ../one-wait-runtime/corelib/dynamicemb
DEMB_ARCHS=90 python setup.py build_ext --inplace
export PYTHONPATH="$PWD:$PWD/benchmark${PYTHONPATH:+:$PYTHONPATH}"

python -m pytest -q --rootdir="$POC" "$POC/check_lookup.py"
python "$POC/benchmark.py" --storage hbm --batch-size 256
python "$POC/benchmark.py" --storage hbm --batch-size 16384 --hotness 256
python "$POC/benchmark.py" --storage caching --batch-size 256 --inputs 1
python "$POC/benchmark.py" --storage caching --batch-size 4096 --inputs 2
```

Each run reports the package and extension paths, median host and wall milliseconds per training step, individual timing windows, and a separate CUDA profile. Inspect `profile_per_step.syncs_per_step` for blocking calls. `cache_hit_rate` comes from a separate pass outside the measurements. The arms rotate on the same warmed tables and start each timed window on the same input. A single input exercises warm hits. Two disjoint inputs and the default small cache exercise eviction and reload.

The experiment uses one training stream, fixed-capacity built-in FP32 tables, SGD, and the benchmark's built-in initializer. Cache admission, custom storage and growth are outside this comparison. The benchmark disables #545's legacy outstanding-key estimate on its own cache modules, as #557 removed that check. Concurrent prefetch requires the atomic pin fix in #557. Cache reservation changes the key map in place. Discard the benchmark module after a placement or initialization error; the PoC does not roll back reservations for reuse.
