from __future__ import annotations

import unittest

import pandas as pd

from scripts.build_final_status import analyze_duplicates


class FinalStatusBuilderTests(unittest.TestCase):
    def test_duplicate_analysis_counts_exact_feature_and_near_duplicates(self) -> None:
        rows = []
        for values, category, target in (
            (0.0, 0, 0),
            (0.0, 0, 1),
            (0.01, 0, 0),
            (1.0, 1, 0),
            (1.0, 1, 0),
        ):
            row = {f"x{i}": values for i in range(5)}
            row.update({f"c{i}": category for i in range(8)})
            row["target"] = target
            rows.append(row)

        result = analyze_duplicates(
            pd.DataFrame(rows),
            [f"x{i}" for i in range(5)],
            [f"c{i}" for i in range(8)],
            "target",
        )

        self.assertEqual(result["exact_duplicate_rows"], 1)
        self.assertEqual(result["exact_unique_rows"], 4)
        self.assertEqual(result["feature_duplicate_rows"], 4)
        self.assertEqual(result["feature_duplicate_groups"], 2)
        self.assertEqual(result["conflicting_target_groups"], 1)
        self.assertEqual(result["near_duplicate_candidate_pairs"], 4)
        self.assertIn("ddof=0", result["standardization"])


if __name__ == "__main__":
    unittest.main()
