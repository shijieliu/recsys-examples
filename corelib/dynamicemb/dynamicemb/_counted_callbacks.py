# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bridge exact-length Python callbacks to the counted lookup interface."""

import inspect
from functools import lru_cache, partial

import torch


@lru_cache(maxsize=128)
def _declares_count(implementation):
    try:
        return "num_valid" in inspect.signature(implementation).parameters
    except (TypeError, ValueError):
        return False


def _accepts_count(callback):
    # Cache the implementation so the lookup does not retain a table instance.
    if inspect.ismethod(callback):
        callback = callback.__func__
    elif not (inspect.isroutine(callback) or isinstance(callback, partial)):
        callback = type(callback).__call__
    return _declares_count(callback)


def _length(count, capacity):
    return capacity if count is None else max(0, min(int(count), capacity))


def _prefix(value, n):
    return None if value is None else value[:n]


def _pad(value, capacity):
    if value is None or value.size(0) == capacity:
        return value
    result = value.new_empty((capacity, *value.shape[1:]))
    result[: value.size(0)].copy_(value)
    return result


def find_counted(
    storage,
    keys,
    table_ids,
    copy_mode,
    lfu_accumulated_frequency=None,
    key_scores=None,
    *,
    num_valid
):
    capacity = keys.numel()
    options = {"copy_mode": copy_mode}
    if _accepts_count(storage.find):
        # Built-in find preserves its exact-length public result when called
        # without a count. Lookup needs its capacity-sized device result.
        if num_valid is None:
            num_valid = keys.new_full((1,), capacity, dtype=torch.int64)
        options["num_valid"] = num_valid
    else:
        n = _length(num_valid, capacity)
        keys, table_ids = keys[:n], table_ids[:n]
        lfu_accumulated_frequency = _prefix(lfu_accumulated_frequency, n)
        key_scores = _prefix(key_scores, n)
    if lfu_accumulated_frequency is not None:
        options["lfu_accumulated_frequency"] = lfu_accumulated_frequency
    if key_scores is not None:
        options["key_scores"] = key_scores
    result = storage.find(keys, table_ids, **options)
    if "num_valid" in options:
        return result
    count, *values = result
    if isinstance(count, torch.Tensor):
        count = count.to(device=keys.device, dtype=torch.int64).reshape(1)
    else:
        count = keys.new_full((1,), count, dtype=torch.int64)
    return (count, *(_pad(value, capacity) for value in values))


def insert_counted(
    storage, keys, table_ids, values, scores=None, *, num_valid, preserve_existing=False
):
    options = {}
    if _accepts_count(storage.insert):
        options["num_valid"] = num_valid
    else:
        n = _length(num_valid, keys.numel())
        if not n:
            return
        keys, table_ids, values = keys[:n], table_ids[:n], values[:n]
        scores = _prefix(scores, n)
    if preserve_existing:
        options["preserve_existing"] = True
    if scores is None:
        return storage.insert(keys, table_ids, values, **options)
    return storage.insert(keys, table_ids, values, scores, **options)


def admit_counted(admitter, keys, table_ids, frequency, *, num_valid):
    if _accepts_count(admitter.admit):
        return admitter.admit(keys, table_ids, frequency, num_valid=num_valid)
    capacity = keys.numel()
    n = _length(num_valid, capacity)
    if not n:
        return keys.new_empty(capacity, dtype=torch.bool)
    mask = admitter.admit(keys[:n], table_ids[:n], _prefix(frequency, n))
    return _pad(mask, capacity)


def initialize_counted(
    initializer, buffer, keys, table_ids, indices, *, num_valid, num_rows=None
):
    if _accepts_count(initializer):
        return initializer(buffer, keys, table_ids, indices, num_valid=num_valid)
    # indices select rows of buffer; keys and table_ids describe the entire
    # live buffer, including rows this call does not initialize.
    if indices is None:
        rows = _length(num_valid, buffer.size(0))
        if not rows:
            return
        indices = torch.arange(rows, dtype=torch.int64, device=buffer.device)
    else:
        if num_valid is not None and num_rows is not None:
            lengths = torch.cat((num_valid.reshape(1), num_rows.reshape(1))).tolist()
            n = _length(lengths[0], indices.size(0))
            rows = _length(lengths[1], buffer.size(0))
        else:
            n = _length(num_valid, indices.size(0))
            rows = _length(num_rows, buffer.size(0))
        if not n:
            return
        indices = indices[:n]
    return initializer(
        buffer[:rows], _prefix(keys, rows), _prefix(table_ids, rows), indices
    )
