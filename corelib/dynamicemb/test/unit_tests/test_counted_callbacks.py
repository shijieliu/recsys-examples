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

import unittest

import torch
from dynamicemb._counted_callbacks import (
    admit_counted,
    find_counted,
    initialize_counted,
    insert_counted,
)


class TestCountedCallbacks(unittest.TestCase):
    def test_exact_storage_gets_live_keys_and_returns_capacity_outputs(self):
        class ExactStorage:
            def find(self, keys, tids, copy_mode, lfu_accumulated_frequency=None):
                self.keys_seen = keys.tolist()
                return (
                    1,
                    keys[:1],
                    torch.tensor([0]),
                    tids[:1],
                    None,
                    torch.tensor([False, True]),
                    keys.clone(),
                    keys[:, None],
                )

            def insert(self, keys, tids, values, scores=None, preserve_existing=False):
                self.inserted = (
                    keys.tolist(),
                    tids.tolist(),
                    values.tolist(),
                    scores.tolist(),
                    preserve_existing,
                )

        store = ExactStorage()
        keys = torch.tensor([19, 23, -9])
        tids = torch.tensor([1, 0, -9])
        n = torch.tensor([2])
        count, miss, positions, miss_ids, scores, hits, _, values = find_counted(
            store, keys, tids, copy_mode=0, num_valid=n
        )
        self.assertEqual(store.keys_seen, [19, 23])
        self.assertEqual(count.tolist(), [1])
        self.assertEqual([miss[0], positions[0], miss_ids[0]], [19, 0, 1])
        self.assertIsNone(scores)
        self.assertEqual(hits[:2].tolist(), [False, True])
        self.assertEqual(values.shape, (3, 1))
        self.assertEqual(miss.shape, (3,))
        insert_counted(
            store, keys, tids, values, keys, preserve_existing=True, num_valid=n
        )
        self.assertEqual(
            store.inserted, ([19, 23], [1, 0], [[19], [23]], [19, 23], True)
        )

    def test_exact_admission_and_initialization_skip_padding(self):
        class ExactAdmitter:
            def admit(self, keys, tids, frequency):
                self.input = (keys.tolist(), tids.tolist(), frequency.tolist())
                return frequency > 2

        admission = ExactAdmitter()
        keys = torch.tensor([11, 12, 13, -99])
        tids = torch.tensor([0, 1, 2, -99])
        n = torch.tensor([3])
        mask = admit_counted(admission, keys, tids, keys - 10, num_valid=n)
        self.assertEqual(admission.input, ([11, 12, 13], [0, 1, 2], [1, 2, 3]))
        self.assertEqual(mask[:3].tolist(), [False, False, True])
        self.assertEqual(mask.shape, keys.shape)
        out = torch.zeros(4, 1, dtype=torch.int64)

        def initialize(buffer, keys, tids, indices):
            self.assertEqual(len(keys), len(buffer))
            self.assertTrue((tids >= 0).all())
            buffer[indices, 0] = keys[indices] + tids[indices]

        initialize_counted(
            initialize,
            out,
            keys,
            tids,
            torch.tensor([2, 1, -99]),
            num_valid=torch.tensor([2]),
            num_rows=n,
        )
        self.assertEqual(out.flatten().tolist(), [0, 13, 15, 0])
        initialize_counted(initialize, out, keys, tids, None, num_valid=n)
        self.assertEqual(out.flatten().tolist(), [11, 13, 15, 0])

    def test_counted_admission_receives_capacity_and_original_count(self):
        keys = torch.tensor([1, -99])
        count = torch.tensor([1])

        class CountedAdmitter:
            def admit(self, incoming, tids, frequency, *, num_valid):
                assert incoming is keys and num_valid is count
                return incoming > 0

        self.assertEqual(
            admit_counted(
                CountedAdmitter(), keys, keys, None, num_valid=count
            ).tolist(),
            [True, False],
        )
