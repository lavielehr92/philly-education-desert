from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# -----------------------------
# Philly constants
# -----------------------------
STATE = "42"           # Pennsylvania
COUNTY = "101"         # Philadelphia County
ACS_YEAR_DEFAULT = 2023  # ACS 5-year 2019-2023
AVAILABLE_YEARS: Sequence[int] = tuple(range(2014, 2024))

CENSUS_BASE = "https://api.census.gov/data"
TIGER_BASE = "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/Tracts_Blocks/MapServer"

# Cornerstone sites (will geocode if lat/lon not provided)
CCA_SITES = [
    {"name": "Cornerstone (58th St)", "address": "939 S. 58th St. Philadelphia, PA"},
    {"name": "Cornerstone (Baltimore Ave)", "address": "4109 Baltimore Ave Philadelphia, PA"},
]

# -----------------------------
# Data model
# -----------------------------
@dataclass(frozen=True)
class Variable:
    alias: str
    table: str
    ids: Sequence[str]  # list of detail cell ids to fetch (e.g., ["B15003_001E", ...])


# NOTES (block-group-available detailed tables):
# B15003 = Educational Attainment (25+) - lets us compute <HS and Bachelor's+
# B28002 = Internet Subscription in Household - total, "No internet"
# B08201 = Vehicles Available - total, "No vehicles"
# B09001 = Population under 18 years (in households)
# B01003 = Total population
# B19013 = Median Household Income (available at block group, but noisy; keep with MOE caveat)

VARIABLES: Sequence[Variable] = (
    # B15003: Totals and components to compute %<HS and %BA+
    Variable("b15003_total", "B15003", ["B15003_001E"]),
    Variable("b15003_lt_hs", "B15003",
             [f"B15003_{i:03d}E" for i in range(2, 17)]),  # 2..16 => < high school diploma
    Variable("b15003_bach_plus", "B15003",
             [f"B15003_{i:03d}E" for i in range(21, 25)]),  # 21..24 => Bachelor's or higher

    # Internet
    Variable("b28002_tot_hh", "B28002", ["B28002_001E"]),
    Variable("b28002_no_inet", "B28002", ["B28002_013E"]),

    # Vehicles
    Variable("b08201_tot_hh", "B08201", ["B08201_001E"]),
    Variable("b08201_no_vehicle", "B08201", ["B08201_002E"]),

    # Children & Total Pop
    Variable("b09001_u18", "B09001", ["B09001_001E"]),   # under 18 in households
    Variable("b01003_total_pop", "B01003", ["B01003_001E"]),

    # Income (keep; may be sparse or noisy at BG level)
    Variable("b19013_mhhinc", "B19013", ["B19013_001E"]),
)

# -----------------------------
# Helpers
# -----------------------------
def _call_acs5_blockgroups(year: int, var_ids: Sequence[str], api_key: Optional[str]) -> pd.DataFrame:
    """Fetch ACS 5-year variables for all block groups in Philadelphia County, PA."""
    params: Dict[str, str] = {
        "get": ",".join(["NAME", *var_ids]),
        "for": "block group:*",
        "in": f"state:{STATE} county:{COUNTY} tract:*",
    }
    if api_key:
        params["key"] = api_key
    url = f"{CENSUS_BASE}/{year}/acs/acs5"
    r = requests.get(url, params=params, timeout=90)
    if r.status_code != 200:
        raise RuntimeError(f"ACS error {r.status_code}: {r.text}")
    arr = r.json()
    df = pd.DataFrame(arr[1:], columns=arr[0])
    # standard FIPS strings & GEOIDs
    df["state"] = df["state"].astype(str).str.zfill(2)
    df["county"] = df["county"].astype(str).str.zfill(3)
    df["tract"] = df["tract"].astype(str).str.zfill(6)
    df["block group"] = df["block group"].astype(str).str.zfill(1)
    df["geoid_bg"] = df["state"] + df["county"] + df["tract"] + df["block group"]
    return df


def _sum_cols(df: pd.DataFrame, cols: Sequence[str]) -> pd.Series:
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[cols].sum(axis=1, skipna=True)


def compute_metrics(raw: pd.DataFrame) -> pd.DataFrame:
    f = raw.copy()

    # Ensure numeric
    for col in raw.columns:
        if col.endswith("E"):
            f[col] = pd.to_numeric(f[col], errors="coerce")

    # Build core percents
    f["pct_lt_hs"] = np.where(f["B15003_001E"] > 0,
                              _sum_cols(f, [f"B15003_{i:03d}E" for i in range(2, 17)]) / f["B15003_001E"] * 100,
                              np.nan)

    f["pct_bach_plus"] = np.where(f["B15003_001E"] > 0,
                                  _sum_cols(f, [f"B15003_{i:03d}E" for i in range(21, 25)]) / f["B15003_001E"] * 100,
                                  np.nan)

    f["pct_no_inet"] = np.where(f["B28002_001E"] > 0, f["B28002_013E"] / f["B28002_001E"] * 100, np.nan)
    f["pct_no_vehicle"] = np.where(f["B08201_001E"] > 0, f["B08201_002E"] / f["B08201_001E"] * 100, np.nan)
    f["pct_children"] = np.where(f["B01003_001E"] > 0, f["B09001_001E"] / f["B01003_001E"] * 100, np.nan)

    # Median household income in $000s for display
    f["mhhinc_k"] = pd.to_numeric(f["B19013_001E"], errors="coerce") / 1000.0

    # Z-score helper
    def z(series: pd.Series, invert: bool = False) -> pd.Series:
        vals = pd.to_numeric(series, errors="coerce")
        mu, sigma = vals.mean(), vals.std(ddof=0)
        out = pd.Series(0.0, index=vals.index) if (sigma == 0 or np.isnan(sigma)) else (vals - mu) / sigma
        return -out if invert else out

    # Composite scores (tunable weights)
    f["need_score"] = z(f["pct_lt_hs"]) + z(f["pct_children"])
    f["choice_gap_score"] = z(f["pct_lt_hs"]) - z(f["pct_bach_plus"])
    f["access_friction_score"] = z(f["pct_no_vehicle"]) + z(f["pct_no_inet"]) + z(f["mhhinc_k"], invert=True)

    f["edi_raw"] = (f["need_score"] + f["choice_gap_score"] + f["access_friction_score"]) / 3.0
    # Scale to 0–100
    minv, maxv = f["edi_raw"].min(), f["edi_raw"].max()
    if pd.isna(minv) or pd.isna(maxv) or minv == maxv:
        f["edi_scaled"] = 50.0
    else:
        f["edi_scaled"] = (f["edi_raw"] - minv) / (maxv - minv) * 100.0

    # Simple 3-tier label
    try:
        f["edi_tier"] = pd.qcut(f["edi_scaled"], q=3, labels=["Lower", "Moderate", "Higher"])
    except ValueError:
        f["edi_tier"] = "Higher"

    return f


# -----------------------------
# TIGERweb GeoJSON for PHILLY block groups
# -----------------------------
@lru_cache(maxsize=None)
def _get_tiger_layer_id(layer_name_contains: str = "Block Groups") -> int:
    r = requests.get(TIGER_BASE, params={"f": "json"}, timeout=30)
    r.raise_for_status()
    info = r.json()
    for lyr in info.get("layers", []):
        if layer_name_contains.lower() in lyr.get("name", "").lower():
            return int(lyr["id"])
    raise RuntimeError("Could not find TIGERweb Block Groups layer id.")

@st.cache_data(show_spinner=False)
def fetch_philly_bg_geojson() -> Dict:
    layer_id = _get_tiger_layer_id("Block Groups")
    # STATE and COUNTY fields use 2/3-digit strings, TRact 6, BLKGRP 1
    where = f"STATE='{STATE}' AND COUNTY='{COUNTY}'"
    params = {
        "where": where,
        "outFields": "STATE,COUNTY,TRACT,BLOCK_GROUP",
        "outSR": "4326",
        "f": "geojson"
    }
    url = f"{TIGER_BASE}/{layer_id}/query"
    r = requests.get(url, params=params, timeout=90)
    if r.status_code != 200:
        raise RuntimeError(f"TIGER error {r.status_code}: {r.text}")
    gj = r.json()
    # compute GEOID property to match our ACS join key
    for feat in gj.get("features", []):
        p = feat.get("properties", {})
        state = str(p.get("STATE", "")).zfill(2)
        county = str(p.get("COUNTY", "")).zfill(3)
        tract = str(p.get("TRACT", "")).zfill(6)
        bg = str(p.get("BLOCK_GROUP", "")).zfill(1)
        p["GEOID"] = state + county + tract + bg
    return gj


# -----------------------------
# Geocoding (Census Geocoder)
# -----------------------------
def geocode_one_line(address: str) -> Optional[Tuple[float, float]]:
    try:
        r = requests.get(
            "https://geocoding.geo.census.gov/geocoder/locations/onelineaddress",
            params={"address": address, "benchmark": "Public_AR_Census2020", "format": "json"},
            timeout=20,
        )
        if r.status_code != 200:
            return None
        js = r.json()
        matches = js.get("result", {}).get("addressMatches", [])
        if not matches:
            return None
        loc = matches[0]["coordinates"]
        return float(loc["y"]), float(loc["x"])  # (lat, lon)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def get_site_points(sites: List[Dict[str, str]]) -> pd.DataFrame:
    rows = []
    for s in sites:
        latlon = geocode_one_line(s["address"])
        rows.append({"name": s["name"], "address": s["address"],
                     "lat": latlon[0] if latlon else np.nan,
                     "lon": latlon[1] if latlon else np.nan})
    return pd.DataFrame(rows)


# -----------------------------
# Pull + assemble ACS
# -----------------------------
@st.cache_data(show_spinner=True, ttl=86400)
def load_acs_bg(year: int, api_key: Optional[str]) -> pd.DataFrame:
    # Fetch all variables in one call per table
    table_to_vars: Dict[str, List[str]] = {}
    for v in VARIABLES:
        for col in v.ids:
            table_to_vars.setdefault(v.table, []).append(col)

    merged: Optional[pd.DataFrame] = None
    for table, cols in table_to_vars.items():
        df = _call_acs5_blockgroups(year, cols, api_key)
        if merged is None:
            merged = df
        else:
            # Only keep the id columns & the new measures (drop duplicate NAME)
            merged = merged.merge(df.drop(columns=["NAME"]), on=["state", "county", "tract", "block group", "geoid_bg"], how="left")

    assert merged is not None
    return compute_metrics(merged)

# -----------------------------
# UI
# -----------------------------
def render_map(df: pd.DataFrame, sites_df: pd.DataFrame) -> None:
    geojson = fetch_philly_bg_geojson()

    fig = px.choropleth(
        df,
        geojson=geojson,
        locations="geoid_bg",
        featureidkey="properties.GEOID",
        color="edi_scaled",
        hover_data={
            "NAME": True,
            "pct_lt_hs": ':.1f',
            "pct_bach_plus": ':.1f',
            "pct_no_vehicle": ':.1f',
            "pct_no_inet": ':.1f',
            "pct_children": ':.1f',
            "mhhinc_k": ':.1f',
        },
        color_continuous_scale="YlOrRd",
        labels={"edi_scaled": "Education Desert Index"},
        scope=None,
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))

    # Add Cornerstone markers
    if not sites_df.empty:
        fig.add_scattergeo(
            lat=sites_df["lat"], lon=sites_df["lon"],
            text=sites_df["name"],
            mode="markers+text",
            textposition="top center",
            marker=dict(size=10, symbol="star", line=dict(width=1)),
            name="Cornerstone",
        )
    st.plotly_chart(fig, use_container_width=True)


def render_cards(df: pd.DataFrame) -> None:
    st.subheader("Top Education-Desert Block Groups (Philadelphia)")
    top = df.sort_values("edi_scaled", ascending=False).head(10)[
        ["NAME", "edi_scaled", "pct_lt_hs", "pct_bach_plus", "pct_no_vehicle", "pct_no_inet", "pct_children", "mhhinc_k"]
    ]
    st.dataframe(top.rename(columns={
        "NAME": "Block Group",
        "edi_scaled": "EDI (0–100)",
        "pct_lt_hs": "% < HS",
        "pct_bach_plus": "% Bachelor's+",
        "pct_no_vehicle": "% HHs No Vehicle",
        "pct_no_inet": "% HHs No Internet",
        "pct_children": "% < 18",
        "mhhinc_k": "Median HH Income ($k)",
    }), use_container_width=True)


def render_download(df: pd.DataFrame) -> None:
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download block-group dataset (CSV)", data=csv_bytes,
                       file_name="philly_education_desert_blockgroups.csv", mime="text/csv")


def main() -> None:
    st.set_page_config(page_title="Philadelphia Education Desert (Block Groups)", layout="wide")
    st.title("Philadelphia Education Desert Dashboard — Block-Group View")
    st.caption("ACS 5-year detailed tables at block-group level; composite index is relative within Philadelphia.")

    with st.sidebar:
        st.header("Controls")
        year = st.selectbox("ACS 5-year vintage", options=AVAILABLE_YEARS,
                            index=AVAILABLE_YEARS.index(ACS_YEAR_DEFAULT))
        # prefer pulling from Streamlit secrets if provided
        api_key_default = st.secrets.get("CENSUS_API_KEY", os.getenv("CENSUS_API_KEY", ""))
        api_key = st.text_input("Census API key", value=api_key_default, type="password",
                                help="Paste your U.S. Census API key here. It increases rate limits.")

    # Load data
    df = load_acs_bg(year, api_key if api_key else None)
    sites = get_site_points(CCA_SITES)

    # Optional filter by EDI tier
    tier = st.sidebar.multiselect("Show tiers", ["Higher", "Moderate", "Lower"], default=["Higher", "Moderate", "Lower"])
    view = df[df["edi_tier"].astype(str).isin(tier)].copy()

    # Map + summary
    render_map(view, sites)
    render_cards(view)
    render_download(view)

    # Methodology + caveats
    st.markdown("""
**Methodology (block-group level)**
- **Need**: % adults 25+ with less than HS (**B15003** cells 2–16 / 001) and % of population under 18 (**B09001_001E** / **B01003_001E**).
- **Choice Gap**: high %<HS offset by low %Bachelor’s+ (**B15003** cells 21–24 / 001).
- **Access Friction**: % HHs without a vehicle (**B08201_002E** / **001E**), % HHs without Internet (**B28002_013E** / **001E**), and **median HH income** (B19013_001E, inverted).
- **Composite (EDI)**: z-scored sub-pillars averaged, scaled 0–100, ranked within **Philadelphia County** only.

**Caveats**
- Block-group medians (income) and small counts have higher sampling error. Interpret small differences cautiously; use tiers rather than exact ranks.
- Internet/vehicle are **household** measures; education attainment is **persons 25+**. The mix reflects conditions relevant to K-8 families’ context but is not a direct school-quality metric.
""")

    st.markdown("**Interpretation prompts for Cornerstone**")
    st.write("""
- **Recruitment focus**: Start with the **Higher** EDI block groups within a short travel time to each site. Partner with nearby churches/community orgs.
- **Barrier removal pilot**: In blocks with high `% HHs No Vehicle` or `% HHs No Internet`, test transit vouchers, shuttle pickups, and paper-first or SMS application flows.
- **Messaging**: Where `mhhinc_k` is lowest, lead with **net tuition examples** rather than list price; publish a one-screen calculator.
    """)

if __name__ == "__main__":
    main()
