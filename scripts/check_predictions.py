"""Sanity checks for src/predict.py: example verdicts, explanations, and bad-input handling.

Run:  python scripts/check_predictions.py   (after python src/train.py)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8")  # so ₹ prints on Windows consoles

from config import EXAMPLES  # noqa: E402
from predict import fmt_price, predict  # noqa: E402

failures = 0


def check(ok: bool, message: str):
    global failures
    failures += not ok
    print(("PASS " if ok else "FAIL ") + message)


def show(r):
    print(f"     estimate {fmt_price(r.estimate_lakhs)}, range {fmt_price(r.range_lakhs[0])}-"
          f"{fmt_price(r.range_lakhs[1])}, asking {fmt_price(r.asking_lakhs)} (x{r.ratio:.2f})")
    for f in r.factors:
        print("       -", f["text"])
    for n in r.notes:
        print("       note:", n)


print("Example buttons")
for name, ex in EXAMPLES.items():
    r = predict(ex["listing"], ex["asking_lakhs"])
    check(r.verdict == ex["expected"], f"{name}: {r.verdict} (expected {ex['expected']})")
    check(len(r.factors) >= 3, f"{name}: {len(r.factors)} explanation factors")
    show(r)

base = EXAMPLES["Try a fair listing"]["listing"]

print("\nHand-made listings across the bands")
for asking, expected in [(30, "Suspicious"), (45, "Good Deal"), (65, "Fairly Priced"), (100, "Overpriced"), (150, "Suspicious")]:
    r = predict(base, asking)
    check(r.verdict == expected, f"Whitefield 2 BHK at {fmt_price(asking)}: {r.verdict} (expected {expected})")

print("\nOther property types")
for desc, listing, asking in [
    ("Plot, 4 BHK, 2400 sq ft, Hebbal", dict(base, location="Hebbal", area_type="Plot", area_sqft=2400, bhk=4, bathrooms=4, balconies=2), 300),
    ("1 BHK, 600 sq ft, under construction", dict(base, area_sqft=600, bhk=1, bathrooms=1, ready_to_move=0), 30),
]:
    r = predict(listing, asking)
    check(len(r.factors) >= 3, f"{desc}: {r.verdict}")
    show(r)

print("\nUnknown locality falls back to 'other'")
r = predict(dict(base, location="Atlantis Nagar"), 60)
check(bool(r.notes), f"Atlantis Nagar: {r.verdict}, note shown")
show(r)

print("\nBad inputs raise friendly errors")
for desc, listing, asking in [
    ("negative area", dict(base, area_sqft=-5), 60),
    ("zero BHK", dict(base, bhk=0), 60),
    ("zero asking price", base, 0),
    ("missing bathrooms", {k: v for k, v in base.items() if k != "bathrooms"}, 60),
]:
    try:
        predict(listing, asking)
        check(False, f"{desc}: no error raised")
    except ValueError as e:
        check(True, f"{desc}: {e}")

print("\nCramped listing gives a warning, not a crash")
r = predict(dict(base, area_sqft=500, bhk=3), 40)
check(bool(r.notes), f"500 sq ft 3 BHK: {r.notes[0] if r.notes else 'no warning'}")

print(f"\n{'All checks passed' if not failures else f'{failures} check(s) FAILED'}")
sys.exit(1 if failures else 0)
