# SPDX-License-Identifier: Apache-2.0
"""Retained STORE candidate selection leaves plan-held obligations untouched."""
# Standard
from collections import OrderedDict
import unittest

# First Party
from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector


class RetainedStorePlanContract(unittest.TestCase):
    def setUp(self):
        # No worker or transport is needed for this scalar selection boundary.
        self.connector = LMCacheMPConnector.__new__(LMCacheMPConnector)
        self.connector.operation_step_limit = 1
        self.connector._retained_store_candidates = OrderedDict(
            (identity, None) for identity in ("hold", "send", "later")
        )
        self.connector._store_operation_views = {
            identity: {"operation_id": identity}
            for identity in self.connector._retained_store_candidates
        }

    def test_hold_does_not_use_another_store_visit_or_change_its_position(self):
        for preferred in ((), ("hold", "send")):
            with self.subTest(preferred=preferred):
                self.setUp()
                result = self.connector.get_retained_store_candidates(
                    1, preferred_ids=preferred, excluded_ids=("hold",)
                )
                self.assertEqual(result, ({"operation_id": "send"},))
                self.assertEqual(list(self.connector._retained_store_candidates),
                                 ["hold", "later", "send"])
                self.assertEqual(len(self.connector._store_operation_views), 3)

    def test_no_exclusions_preserves_native_round_robin(self):
        selected = [self.connector.get_retained_store_candidates(1)[0]["operation_id"]
                    for _ in range(4)]
        self.assertEqual(selected, ["hold", "send", "later", "hold"])


if __name__ == "__main__":
    unittest.main()
