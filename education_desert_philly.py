from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# -----------------------------
# Constants: Philly only
# -----------------------------
STATE = "42"           # Pennsylvania
COUNTY = "101"         # Philadelphia County
ACS_YEAR_DEFAULT = 2023
AVAILABLE_YEARS: Sequence[int] = tuple(range(2014, 2024))

CENSUS_BASE = "https://api.census.gov/data"
TIGER_BASE = "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/Tracts_Blocks/MapServer"

CCA_SITES = [
    {"name": "Cornerstone (58th St)", "address": "939 S. 58th St. Philadelphia, PA"},
    {"name": "Cornerstone (Baltimore Ave)", "address": "4109 Baltimore Ave Philadelphia, PA"},
]

HTTP_HEADERS = {"User-Agent": "Philly-ED-Streamlit/1.0 (contact: dashboard user)"}

# -----------------------------
# Vars (block-group level tables)
# -----------------------------
@dataclass(frozen=True)
class Variable:
    alias: str
    table: str
    ids: Sequence[str]

VARIABLES: Sequence[Variable] = (
    Variable("b15003_total", "B15003", ["B15003_001E"]),
    Variable("b15003_lt_hs", "B15003", [f"B15003_{i:03d}E" for i in range(2, 17)]),
    Variable("b15003_bach_plus", "B15003", [f"B15003_{i:03d}E" for i in range(21, 25)]),
    Variable("b28002_tot_hh", "B28002", ["B28002_001E"]),
    Variable("b28002_no_inet", "B28002", ["B28002_013E"]),
    Variable("b08201_tot_hh", "B08201", ["B08201_001E"]),
    Variable("b08201_no_vehicle", "B08201", ["B08201_002E"]),
    Variable("b09001_u18", "B09001", ["B09001_001E"]),
    Variable("b01003_total_pop", "B01003", ["B01003_001E"]),
    Variable("b19013_mhhinc", "B19013", ["B19013_001E"]),
)

# -----------------------------
# Helpers
# -----------------------------
def _safe_json(resp: requests.Response) -> list:
    """Return resp.json() with clearer errors (handles HTML errors/rate limits)."""
    ct = resp.headers.get("Content-Type", "")
    if not resp.ok:
        raise RuntimeError(f"ACS error {resp.status_code}: {resp.text[:200]}")
    try:
        return resp.json()
    except requests.exceptions.JSONDecodeError:
        snippet = resp.text[:200]
        raise RuntimeError(
            "ACS response was not JSON (often a rate-limit or service hiccup). "
            f"Content-Type={ct!r} Snippet={snippet!r}"
        )

def _call_acs5_blockgroups(year: int, var_ids: Sequence[str], api_key: Optional[str]) -> pd.DataFrame:
    params: Dict[str, str] = {
        "get": ",".join(["NAME", *var_ids]),
        "for": "block group:*",
        "in": f"state:{STATE} county:{COUNTY} tract:*",
    }
    if api_key:
        params["key"] = api_key
    url = f"{CENSUS_BASE}/{year}/acs/acs5"
    r = requests.get(url, params=params, headers=HTTP_HEADERS, timeout=90)
    arr = _safe_json(r)
    if not arr or len(arr) < 2:
        raise RuntimeError("ACS returned no rows for Philadelphia block groups.")
    df = pd.DataFrame(arr[1:], columns=arr[0])
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
    for col in raw.columns:
        if col.endswith("E"):
            f[col] = pd.to_numeric(f[col], errors="coerce")

    f["pct_lt_hs"] = np.where(
        f["B15003_001E"] > 0,
        _sum_cols(f, [f"B15003_{i:03d}E" for i in range(2, 17)]) / f["B15003_001E"] * 100,
        np.nan,
    )
    f["pct_bach_plus"] = np.where(
        f["B15003_001E"] > 0,
        _sum_cols(f, [f"B15003_{i:03d}E" for i in range(21, 25)]) / f["B15003_001E"] * 100,
        np.nan,
    )
    f["pct_no_inet"] = np.where(f["B28002_001E"] > 0, f["B28002_013E"] / f["B28002_001E"] * 100, np.nan)
    f["pct_no_vehicle"] = np.where(f["B08201_001E"] > 0, f["B08201_002E"] / f["B08201_001E"] * 100, np.nan)
    f["pct_children"] = np.where(f["B01003_001E"] > 0, f["B09001_001E"] / f["B01003_001E"] * 100, np.nan)
    f["mhhinc_k"] = pd.to_numeric(f["B19013_001E"], errors="coerce") / 1000.0

    # Clean display values
    for c in ["pct_lt_hs", "pct_bach_plus", "pct_no_inet", "pct_no_vehicle", "pct_children"]:
        f[c] = f[c].clip(lower=0, upper=100)
    f["mhhinc_k"] = f["mhhinc_k"].where(f["mhhinc_k"] > 0)

    # Z-score composite
    def z(s: pd.Series, invert: bool = False) -> pd.Series:
        vals = pd.to_numeric(s, errors="coerce")
        mu, sd = vals.mean(), vals.std(ddof=0)
        out = pd.Series(0.0, index=vals.index) if (sd == 0 or np.isnan(sd)) else (vals - mu) / sd
        return -out if invert else out

    f["need_score"] = z(f["pct_lt_hs"]) + z(f["pct_children"])
    f["choice_gap_score"] = z(f["pct_lt_hs"]) - z(f["pct_bach_plus"])
    f["access_friction_score"] = z(f["pct_no_vehicle"]) + z(f["pct_no_inet"]) + z(f["mhhinc_k"], invert=True)
    f["edi_raw"] = (f["need_score"] + f["choice_gap_score"] + f["access_friction_score"]) / 3.0

    minv, maxv = f["edi_raw"].min(), f["edi_raw"].max()
    f["edi_scaled"] = 50.0 if (pd.isna(minv) or pd.isna(maxv) or minv == maxv) else (f["edi_raw"] - minv) / (maxv - minv) * 100.0

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
    r = requests.get(TIGER_BASE, params={"f": "json"}, headers=HTTP_HEADERS, timeout=30)
    r.raise_for_status()
    info = r.json()
    for lyr in info.get("layers", []):
        if layer_name_contains.lower() in lyr.get("name", "").lower():
            return int(lyr["id"])
    raise RuntimeError("Could not find TIGERweb Block Groups layer id.")

@st.cache_data(show_spinner=False)
def fetch_philly_bg_geojson() -> Dict:
    layer_id = _get_tiger_layer_id("Block Groups")
    where = f"STATE='{STATE}' AND COUNTY='{COUNTY}'"
    params = {"where": where, "outFields": "STATE,COUNTY,TRACT,BLOCK_GROUP", "outSR": "4326", "f": "geojson"}
    url = f"{TIGER_BASE}/{layer_id}/query"
    r = requests.get(url, params=params, headers=HTTP_HEADERS, timeout=90)
    if not r.ok:
        raise RuntimeError(f"TIGER error {r.status_code}: {r.text[:180]}")
    gj = r.json()
    for feat in gj.get("features", []):
        p = feat.get("properties", {})
        p["GEOID"] = str(p.get("STATE", "")).zfill(2) + str(p.get("COUNTY", "")).zfill(3) + \
                     str(p.get("TRACT", "")).zfill(6) + str(p.get("BLOCK_GROUP", "")).zfill(1)
    return gj

# -----------------------------
# Geocoding (Census)
# -----------------------------
def geocode_one_line(address: str) -> Optional[Tuple[float, float]]:
    try:
        r = requests.get(
            "https://geocoding.geo.census.gov/geocoder/locations/onelineaddress",
            params={"address": address, "benchmark": "Public_AR_Census2020", "format": "json"},
            headers=HTTP_HEADERS, timeout=20,
        )
        if not r.ok:
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
# Load ACS
# -----------------------------
@st.cache_data(show_spinner=True, ttl=86400)
def load_acs_bg(year: int, api_key: Optional[str]) -> pd.DataFrame:
    if not api_key:
        raise RuntimeError("No Census API key found. Add it in Settings → Secrets as CENSUS_API_KEY.")
    table_to_vars: Dict[str, List[str]] = {}
    for v in VARIABLES:
        for col in v.ids:
            table_to_vars.setdefault(v.table, []).append(col)

    merged: Optional[pd.DataFrame] = None
    for table, cols in table_to_vars.items():
        df = _call_acs5_blockgroups(year, cols, api_key)
        merged = df if merged is None else merged.merge(
            df.drop(columns=["NAME"]), on=["state", "county", "tract", "block group", "geoid_bg"], how="left"
        )
    assert merged is not None
    return compute_metrics(merged)

# -----------------------------
# UI
# -----------------------------
def render_map(df: pd.DataFrame, sites_df: pd.DataFrame) -> None:
    # Ensure join keys are strings
    df = df.copy()
    df["geoid_bg"] = df["geoid_bg"].astype(str)

    geojson = fetch_philly_bg_geojson()

    # OPTIONAL one-liner to debug joins (uncomment if needed)
    # matched = sum(df["geoid_bg"].isin({f["properties"]["GEOID"] for f in geojson.get("features", [])}))
    # st.caption(f"{matched} / {len(df)} block groups matched to geometry")

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
    )

    # Make blocks visible & crisp
    fig.update_traces(
        selector=dict(type="choropleth"),
        marker_line_width=0.5,
        marker_line_color="black",
        hovertemplate="%{customdata[0]}<br>EDI=%{z:.1f}<extra></extra>",
        opacity=0.9,
    )

    # Fit to Philly geometry and hide the empty globe frame
    fig.update_geos(fitbounds="locations", visible=False)

    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)

    # Cornerstone markers (draw after polygons)
    if not sites_df.empty:
        st.plotly_chart(
            fig.add_scattergeo(
                lat=sites_df["lat"],
                lon=sites_df["lon"],
                text=sites_df["name"],
                mode="markers+text",
                textposition="top center",
                marker=dict(size=10, symbol="star", line=dict(width=1)),
                name="Cornerstone",
            ),
            use_container_width=True,
        )

def render_cards(df: pd.DataFrame) -> None:
    st.subheader("Top Education-Desert Block Groups (Philadelphia)")
    top = df.sort_values("edi_scaled", ascending=False).head(10).copy()
    display = top[["NAME","edi_scaled","pct_lt_hs","pct_bach_plus","pct_no_vehicle","pct_no_inet","pct_children","mhhinc_k"]]
    display = display.rename(columns={
        "NAME":"Block Group","edi_scaled":"EDI (0–100)","pct_lt_hs":"% < HS",
        "pct_bach_plus":"% Bachelor's+","pct_no_vehicle":"% HHs No Vehicle",
        "pct_no_inet":"% HHs No Internet","pct_children":"% < 18","mhhinc_k":"Median HH Income ($k)",
    })
    display = display.round({"edi_scaled":1,"pct_lt_hs":1,"pct_bach_plus":1,"pct_no_vehicle":1,"pct_no_inet":1,"pct_children":1,"mhhinc_k":1})
    st.dataframe(display.fillna(""), use_container_width=True)

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
        # No API key input here—key is secrets-only

    # Read key from secrets/env only
    API_KEY = st.secrets.get("CENSUS_API_KEY", os.getenv("CENSUS_API_KEY", "")) or None

    # Helpful status chip
    st.info("Census key loaded from secrets." if API_KEY else "No Census key found in secrets.", icon="🔑")

    try:
        df = load_acs_bg(year, API_KEY)
    except Exception as e:
        st.error(f"Data load failed: {e}")
        return

    sites = get_site_points(CCA_SITES)
    tier = st.sidebar.multiselect("Show tiers", ["Higher", "Moderate", "Lower"], default=["Higher", "Moderate", "Lower"])
    view = df[df["edi_tier"].astype(str).isin(tier)].copy()

    render_map(view, sites)
    render_cards(view)
    render_download(view)

    st.markdown("""
**Methodology (block-group level)**  
- **Need**: % adults 25+ with <HS (B15003 cells 2–16 / 001) and % population <18 (B09001_001E / B01003_001E).  
- **Choice Gap**: %<HS offset by low %Bachelor’s+ (B15003 cells 21–24 / 001).  
- **Access Friction**: % HHs without a vehicle (B08201_002E / 001E), % HHs without Internet (B28002_013E / 001E), and median HH income (B19013_001E, inverted).  
- **Composite**: z-scored pillars averaged, scaled to 0–100, ranked within **Philadelphia** only.
""")

if __name__ == "__main__":
    main()

