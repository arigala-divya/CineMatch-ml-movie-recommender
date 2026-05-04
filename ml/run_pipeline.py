# ── CineMatch ML Pipeline Runner ──────────────────────────────────────────────
# Run all 3 steps in sequence:
#   Step 1 → Customer Segmentation   (K-Means clustering)
#   Step 2 → Segment Insights        (analytics & visualisations)
#   Step 3 → Recommendation Engine   (hybrid scoring per age group)
#
# Usage:  python ml/run_pipeline.py
# ──────────────────────────────────────────────────────────────────────────────
import subprocess
import sys
from pathlib import Path

ML_DIR = Path(__file__).parent

STEPS = [
    ("Step 1 — Customer Segmentation",  "step1_segmentation.py"),
    ("Step 2 — Segment Insights",       "step2_insights.py"),
    ("Step 3 — Recommendation Engine",  "step3_recommendations.py"),
]

print("\n" + "=" * 60)
print("  CineMatch ML Pipeline")
print("=" * 60)

for label, script in STEPS:
    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"{'─' * 60}")
    result = subprocess.run(
        [sys.executable, str(ML_DIR / script)],
        check=True
    )

print("\n" + "=" * 60)
print("  All steps completed successfully!")
print("  Outputs written to  data/processed/  and  visuals/")
print("=" * 60 + "\n")
