"""IBTrACS loading, storm series, forecast-case construction and truth."""
from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd

from .config import get_paths, work_dir, LEADS_H

USECOLS = [
    "SID", "SEASON", "BASIN", "NAME", "ISO_TIME", "NATURE",
    "LAT", "LON", "USA_WIND", "USA_PRES", "WMO_WIND", "WMO_PRES",
    "DIST2LAND", "USA_SSHS",
]


def load_ibtracs(csv_path: str | None = None, refresh: bool = False) -> pd.DataFrame:
    """Load and clean IBTrACS to 6-hourly synoptic records with USA intensity.

    Results are cached to <work>/cache/ibtracs.pkl because the CSV is ~300 MB.
    """
    csv_path = csv_path or get_paths().ibtracs_csv
    cache = os.path.join(work_dir("cache"), "ibtracs.pkl")
    if os.path.exists(cache) and not refresh:
        with open(cache, "rb") as f:
            return pickle.load(f)

    # keep_default_na off so the North Atlantic basin code "NA" is not read as
    # NaN; numeric columns are coerced explicitly below.
    df = pd.read_csv(csv_path, skiprows=[1], usecols=USECOLS, low_memory=False,
                     keep_default_na=False)
    df["BASIN"] = df["BASIN"].replace("", "NA")
    for c in ["LAT", "LON", "USA_WIND", "USA_PRES", "WMO_WIND", "WMO_PRES", "SEASON", "DIST2LAND"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], errors="coerce")
    df = df.dropna(subset=["ISO_TIME", "LAT", "LON"])
    # synoptic 6-hourly records only
    df = df[df["ISO_TIME"].dt.hour.isin([0, 6, 12, 18]) & (df["ISO_TIME"].dt.minute == 0)]
    # merged intensity: USA agency (1-min kt) preferred, WMO fallback
    df["VMAX"] = df["USA_WIND"].fillna(df["WMO_WIND"])
    df["PRES"] = df["USA_PRES"].fillna(df["WMO_PRES"])
    df = df.sort_values(["SID", "ISO_TIME"]).reset_index(drop=True)
    with open(cache, "wb") as f:
        pickle.dump(df, f, protocol=4)
    return df


def storm_groups(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {sid: g.reset_index(drop=True) for sid, g in df.groupby("SID")}


def _history(g: pd.DataFrame, i: int, steps: int = 4) -> list[dict]:
    """Last `steps` 6-hourly records ending at row i (inclusive); pads with the
    earliest record when the storm is young."""
    rows = []
    for k in range(steps - 1, -1, -1):
        j = max(i - k, 0)
        r = g.iloc[j]
        rows.append(dict(
            time=str(r.ISO_TIME), lat=float(r.LAT), lon=float(r.LON),
            vmax=None if pd.isna(r.VMAX) else float(r.VMAX),
            pres=None if pd.isna(r.PRES) else float(r.PRES),
        ))
    return rows


def build_cases(df: pd.DataFrame, year_lo: int, year_hi: int,
                init_hours=(0, 12), min_vmax: float = 34.0,
                require_next24: bool = True) -> pd.DataFrame:
    """Enumerate forecast cases: storm alive at init with VMAX >= min_vmax.

    Truth columns lat_XX/lon_XX/vmax_XX are NaN when the storm record ends
    before that lead (verification is homogeneous per lead downstream).
    """
    out = []
    for sid, g in df.groupby("SID"):
        season = int(g["SEASON"].iloc[0])
        if season < year_lo or season > year_hi:
            continue
        g = g.reset_index(drop=True)
        times = g["ISO_TIME"]
        index_by_time = {t: i for i, t in enumerate(times)}
        for i in range(len(g)):
            r = g.iloc[i]
            t0 = r.ISO_TIME
            if t0.hour not in init_hours:
                continue
            if pd.isna(r.VMAX) or r.VMAX < min_vmax:
                continue
            truth = {}
            n_valid = 0
            for lead in LEADS_H:
                tv = t0 + pd.Timedelta(hours=lead)
                j = index_by_time.get(tv)
                if j is None:
                    truth[f"lat_{lead}"] = np.nan
                    truth[f"lon_{lead}"] = np.nan
                    truth[f"vmax_{lead}"] = np.nan
                else:
                    rj = g.iloc[j]
                    truth[f"lat_{lead}"] = float(rj.LAT)
                    truth[f"lon_{lead}"] = float(rj.LON)
                    truth[f"vmax_{lead}"] = np.nan if pd.isna(rj.VMAX) else float(rj.VMAX)
                    n_valid += 1
            if require_next24 and np.isnan(truth.get("lat_24", np.nan)):
                continue
            out.append(dict(
                sid=sid, name=str(r.NAME), basin=str(r.BASIN), season=season,
                init=str(t0), lat=float(r.LAT), lon=float(r.LON),
                vmax=float(r.VMAX), pres=np.nan if pd.isna(r.PRES) else float(r.PRES),
                dist2land=np.nan if pd.isna(r.DIST2LAND) else float(r.DIST2LAND),
                nature=str(r.NATURE), n_valid_leads=n_valid,
                history=_history(g, i), **truth,
            ))
    return pd.DataFrame(out)


def crop_dir(sid: str, season: int, time: pd.Timestamp) -> str:
    """Directory of the MSETCD multi-modal crop for a storm timestep."""
    p = get_paths()
    tdir = pd.Timestamp(time).strftime("%Y-%m-%d %H_%M_%S")
    return os.path.join(p.tc_era5, str(season), sid, tdir)
