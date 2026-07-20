"""Build forecast-case tables for all era splits and save to <work>/cases/."""
import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stormdesk.config import TRAIN_YEARS, VAL_YEARS, CALIB_YEARS, TEST_YEARS, work_dir
from stormdesk.ibtracs import load_ibtracs, build_cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    df = load_ibtracs(refresh=args.refresh)
    out = work_dir("cases")
    specs = dict(
        train=(TRAIN_YEARS, (0, 6, 12, 18)),   # dense for DL/CLIPER training
        val=(VAL_YEARS, (0, 6, 12, 18)),
        calib=(CALIB_YEARS, (0,)),  # 00Z only: halves AIWP compute, ~1400 samples suffice for calibration
        test=(TEST_YEARS, (0, 12)),
    )
    for name, ((y0, y1), hours) in specs.items():
        cases = build_cases(df, y0, y1, init_hours=hours)
        with open(os.path.join(out, f"{name}.pkl"), "wb") as f:
            pickle.dump(cases, f, protocol=4)
        print(f"{name}: {len(cases)} cases, {cases['sid'].nunique()} storms, "
              f"seasons {y0}-{y1}")


if __name__ == "__main__":
    main()
