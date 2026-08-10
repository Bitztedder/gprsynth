"""Generate a small dataset and verify it. No survey data required."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gprsynth import dataset, verify
from gprsynth.targets import CLASSES

OUT = os.environ.get("OUT", "demo_dataset")

stat = dataset.make(OUT, n_surveys=40, ntrace=1536, seed=0,
                    progress=lambda i, n: print(f"\r  {i}/{n} surveys", end="", flush=True))
print()
for k, v in stat.items():
    print(f"  {k:12s} {v}")
print()
raise SystemExit(verify.main(OUT, CLASSES))
