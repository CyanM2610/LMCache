# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for explicitly assigned DAX ranges and unchanged LRU legality."""

# Standard
import importlib.util
from pathlib import Path
from types import ModuleType
import sys
import unittest


def load_leaf(name: str) -> ModuleType:
    """Load pure configuration modules without initializing an accelerator."""
    path = (
        Path(__file__).resolve().parents[2] / "lmcache/v1/distributed" / (name + ".py")
    )
    spec = importlib.util.spec_from_file_location("test_" + name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


dax = load_leaf("physical_dax")
config = load_leaf("hybrid_config")
policy = load_leaf("hybrid_policy")
A = 2 << 20


class RegionContracts(unittest.TestCase):
    def test_last_assigned_region_is_legal(self) -> None:
        dax.validate_region(16 * A, A, 12 * A, 4 * A, 2 * A)

    def test_never_exceeds_assignment(self) -> None:
        with self.assertRaises(ValueError):
            dax.validate_region(16 * A, A, 0, 2 * A, 4 * A)

    def test_never_exceeds_device(self) -> None:
        with self.assertRaises(ValueError):
            dax.validate_region(16 * A, A, 14 * A, 4 * A, 2 * A)

    def test_every_range_boundary_is_device_aligned(self) -> None:
        for args in (
            (16 * A, A, 1, 4 * A, A),
            (16 * A, A, 0, 4 * A + 1, A),
            (16 * A, A, 0, 4 * A, A + 1),
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                dax.validate_region(*args)

    def test_no_float_or_empty_ranges(self) -> None:
        for value in (0, -A, float(A), True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                dax.validate_region(16 * A, A, 0, 4 * A, value)

    def test_physical_cxl_requires_explicit_region(self) -> None:
        with self.assertRaises(ValueError):
            config.HybridMemoryConfig(
                0, A, cxl_backend="devdax", dax_path="/dev/dax0.0"
            )

    def test_physical_cxl_rejects_simulator_endpoint(self) -> None:
        with self.assertRaises(ValueError):
            config.HybridMemoryConfig(
                A,
                A,
                cxl_backend="devdax",
                bulk_control="/model",
                dax_path="/dev/dax0.0",
                dax_region_bytes=A,
            )

    def test_dram_only_needs_no_model_or_device(self) -> None:
        self.assertEqual(
            config.HybridMemoryConfig(A, 0, cxl_backend="devdax").cxl_bytes, 0
        )

    def test_nan_timeout_rejected(self) -> None:
        with self.assertRaises(ValueError):
            config.HybridMemoryConfig(A, 0, completion_timeout_s=float("nan"))

    def test_original_policy_and_device_policy_use_same_legal_order(self) -> None:
        for medium in ("cxl_sim", "cxl_dax"):
            candidates = [
                policy.EvictionCandidate("locked", medium, A, 0, False),
                policy.EvictionCandidate("cxl", medium, A, 2, True),
                policy.EvictionCandidate("dram", "dram", A, 1, True),
            ]
            self.assertEqual(
                policy.DramFirstPolicy(medium).victims(candidates), ["cxl"]
            )
            self.assertEqual(
                policy.AllPoolsLRUPolicy(medium).victims(candidates), ["dram", "cxl"]
            )
            self.assertEqual(
                policy.AllPoolsLRUPolicy(medium).pool_order(), ("dram", medium)
            )

    def test_forced_physical_pool_label(self) -> None:
        self.assertEqual(policy.ForcedPoolPolicy("cxl_dax").pool_order(), ("cxl_dax",))


if __name__ == "__main__":
    unittest.main()
