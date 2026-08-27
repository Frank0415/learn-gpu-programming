from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cases import load_manifest


class ManifestTests(unittest.TestCase):
    def test_v1_stage_capability_and_reused_h200_cases(self) -> None:
        manifest = load_manifest(ROOT / "manifests" / "flash_attention_v1.json")
        self.assertEqual(manifest.stage_rules["03c"]["compute_capability"], "sm120")
        self.assertEqual(manifest.stage_rules["03d"]["compute_capability"], "sm90")
        c_ids = {case.case_id for case in manifest.cases if case.stage == "03c"}
        d_ids = {
            case.case_id
            for case in manifest.cases
            if case.stage == "03d" or "03d" in case.also_run_on
        }
        self.assertEqual(c_ids, d_ids)

    def test_v1_covers_required_boundaries_and_gqa_ratios(self) -> None:
        manifest = load_manifest(ROOT / "manifests" / "flash_attention_v1.json")
        lengths = {length for case in manifest.cases for length in case.lengths}
        self.assertTrue({1, 63, 64, 65, 127, 128, 129}.issubset(lengths))
        ratios = {
            case.hq // case.hkv for case in manifest.cases if case.stage == "03c"
        }
        self.assertEqual(ratios, {1, 2, 4, 8})
        scales = {case.scale for case in manifest.cases}
        self.assertTrue({None, 0.0, 1.0}.issubset(scales))

    def test_precision_manifest_keeps_h200_nvfp4_explicit(self) -> None:
        manifest = json.loads(
            (ROOT / "manifests" / "flash_attention_precision_v2.json").read_text()
        )
        h200_nvfp4 = [
            case
            for case in manifest["format_cases"]
            if case["architecture"] == "sm90" and case["format"] == "nvfp4"
        ]
        self.assertEqual(len(h200_nvfp4), 1)
        self.assertEqual(h200_nvfp4[0]["support"], "reference")
        self.assertTrue(manifest["no_silent_fallback"])

    def test_gau_benchmark_budget_is_fixed(self) -> None:
        manifest = load_manifest(ROOT / "manifests" / "flash_attention_v1.json")
        self.assertTrue(
            all(
                (case.warmup_ms, case.rep_ms) == (25, 100)
                for case in manifest.cases
            )
        )
        timing = manifest.performance_policy["timing"]
        self.assertEqual(timing["method"], "triton.testing.do_bench")
        self.assertEqual(timing["return_mode"], "median")
        self.assertTrue(timing["clear_l2"])


if __name__ == "__main__":
    unittest.main()
