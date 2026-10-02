"""Run the Kaggle Facebook bidding A/B test dataset through the repo's ab_test.

Dataset: bombabomba.com Facebook bidding experiment (maximum bidding =
control, average bidding = test), one month of daily aggregate metrics.
Mirrors the feature engineering of the Kaggle notebook
`babyoda/a-b-testing-in-practice` and applies the Shapiro -> t-test/MWU
decision tree from otto_rec.experimentation.hypothesis.

Outputs reports/kaggle_ab_report.json (project) and prints a markdown draft.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from otto_rec.experimentation.hypothesis import ab_test

DATA = Path(r"C:\Users\youss\AppData\Local\Temp\opencode\abdata\ab_testing.xlsx")
OUT_JSON = Path("reports/kaggle_ab_report.json")

METRICS = [
    "Impression",
    "Click",
    "Purchase",
    "Earning",
    "Conversion Rate",
    "Earning per Purchase",
]


def main() -> int:
    control = pd.read_excel(DATA, sheet_name="Control Group")
    test = pd.read_excel(DATA, sheet_name="Test Group")

    for frame in (control, test):
        frame["Conversion Rate"] = frame["Purchase"] / frame["Click"] * 100
        frame["Earning per Purchase"] = frame["Earning"] / frame["Purchase"] * 100

    report: dict = {
        "dataset": {
            "path": str(DATA),
            "control_arm": "maximum bidding (Control Group)",
            "treatment_arm": "average bidding (Test Group)",
            "period": "1 month of daily aggregates",
            "n_days_control": int(len(control)),
            "n_days_treatment": int(len(test)),
            "source": "Kaggle: facebook-bidding-ab-test (mirror of ab_testing_data.xlsx)",
        },
        "hypothesis": {
            "h0": "no statistically significant difference between control and treatment",
            "ha": "there is a statistically significant difference between control and treatment",
            "alpha": 0.05,
            "primary_metric": "Purchase",
        },
        "metrics": {},
    }

    lines: list[str] = []
    for metric in METRICS:
        result = ab_test(control[metric].to_numpy(dtype=float), test[metric].to_numpy(dtype=float))
        report["metrics"][metric] = result
        lines.append(
            f"| {metric} | {result['control']['mean']:.4g} | {result['treatment']['mean']:.4g} | "
            f"{result['normality'].get('both_normal')} | {result['test']} | "
            f"{result['p_value']:.4g} | {'REJECT H0' if result['reject_null'] else 'fail to reject'} |"
        )

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2))
    print("| Metric | Control mean | Treat mean | Normal? | Test | p | Decision |")
    print("|---|---|---|---|---|---|---|")
    print("\n".join(lines))
    print(f"\nwrote {OUT_JSON}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
