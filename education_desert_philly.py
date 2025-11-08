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
    # Additional metrics: poverty, unemployment, renters
    Variable("b17001_poverty_denom", "B17001", ["B17001_001E"]),
    Variable("b17001_poverty_num", "B17001", ["B17001_002E"]),
    Variable("b23025_lf_total", "B23025", ["B23025_001E"]),
    Variable("b23025_unemployed", "B23025", ["B23025_005E"]),
    Variable("b25003_tenure_total", "B25003", ["B25003_001E"]),
    Variable("b25003_renter", "B25003", ["B25003_003E"]),
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
    # Additional metrics
    f["pct_poverty"] = np.where(f["B17001_001E"] > 0, f["B17001_002E"] / f["B17001_001E"] * 100, np.nan)
    f["pct_unemployed"] = np.where(f["B23025_001E"] > 0, f["B23025_005E"] / f["B23025_001E"] * 100, np.nan)
    f["pct_renter"] = np.where(f["B25003_001E"] > 0, f["B25003_003E"] / f["B25003_001E"] * 100, np.nan)

    # Clean display values
    for c in ["pct_lt_hs", "pct_bach_plus", "pct_no_inet", "pct_no_vehicle", "pct_children", 
              "pct_poverty", "pct_unemployed", "pct_renter"]:
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
def render_map(df: pd.DataFrame, sites_df: pd.DataFrame, 
               metric: str = "edi_scaled", 
               color_scale: str = "YlOrRd",
               opacity: float = 0.7,
               basemap_style: str = "carto-positron") -> None:
    """Render choropleth map with configurable metric, color scale, and opacity."""
    df = df.copy()
    df["geoid_bg"] = df["geoid_bg"].astype(str)

    geojson = fetch_philly_bg_geojson()

    # --- Make sure every feature has a top-level id for Plotly to join on ---
    feats = geojson.get("features", [])
    for f in feats:
        if "id" not in f or not f["id"]:
            props = f.get("properties", {})
            geoid = (
                str(props.get("GEOID"))
                or (
                    str(props.get("STATE", "")).zfill(2)
                    + str(props.get("COUNTY", "")).zfill(3)
                    + str(props.get("TRACT", "")).zfill(6)
                    + str(props.get("BLOCK_GROUP", "")).zfill(1)
                    if {"STATE", "COUNTY", "TRACT", "BLOCK_GROUP"} <= set(props.keys())
                    else ""
                )
            )
            f["id"] = geoid

    # Diagnostic: show how many GEOIDs will render
    geo_ids = {f.get("id", "") for f in feats}
    match_count = int(df["geoid_bg"].isin(geo_ids).sum())
    st.caption(f"🗺️ Geometry match: {match_count} / {len(df)} block groups")

    # Prepare custom hover data with all metrics
    hover_data_dict = {
        "NAME": False,
        "geoid_bg": False,
        metric: ':.1f',
        "pct_lt_hs": ':.1f',
        "pct_bach_plus": ':.1f',
        "pct_no_vehicle": ':.1f',
        "pct_no_inet": ':.1f',
        "pct_children": ':.1f',
        "pct_poverty": ':.1f',
        "pct_unemployed": ':.1f',
        "pct_renter": ':.1f',
        "mhhinc_k": ':.1f',
    }
    
    # Metric label mapping
    metric_labels = {
        "edi_scaled": "Education Desert Index",
        "pct_lt_hs": "% < HS",
        "pct_bach_plus": "% Bachelor's+",
        "pct_no_vehicle": "% HHs No Vehicle",
        "pct_no_inet": "% HHs No Internet",
        "pct_children": "% < 18",
        "pct_poverty": "% Poverty",
        "pct_unemployed": "% Unemployed",
        "pct_renter": "% Renter",
        "mhhinc_k": "Median HH Income ($k)",
    }

    fig = px.choropleth(
        df,
        geojson=geojson,
        locations="geoid_bg",
        featureidkey="id",                 # join using the top-level id we enforced above
        color=metric,
        custom_data=["NAME", "geoid_bg", metric, "pct_lt_hs", "pct_bach_plus", "pct_no_vehicle", 
                     "pct_no_inet", "pct_children", "pct_poverty", "pct_unemployed", "pct_renter", "mhhinc_k"],
        hover_data=hover_data_dict,
        color_continuous_scale=color_scale,
        labels={metric: metric_labels.get(metric, metric)},
    )

    # Crisp polygon borders + readable hover
    if fig.data:
        fig.data[0].marker.line.width = 0.6
        fig.data[0].marker.line.color = "black"
        fig.data[0].marker.opacity = opacity
        # Enhanced hover template with all metrics
        fig.data[0].hovertemplate = (
            "<b>%{customdata[0]}</b><br>"
            f"<b>{metric_labels.get(metric, metric)}=%{{customdata[2]:.1f}}</b><br>"
            "<br><i>Education Metrics:</i><br>"
            "% < HS: %{customdata[3]:.1f}%<br>"
            "% Bachelor's+: %{customdata[4]:.1f}%<br>"
            "<br><i>Access Metrics:</i><br>"
            "% No Vehicle: %{customdata[5]:.1f}%<br>"
            "% No Internet: %{customdata[6]:.1f}%<br>"
            "<br><i>Demographics:</i><br>"
            "% < 18: %{customdata[7]:.1f}%<br>"
            "% Poverty: %{customdata[8]:.1f}%<br>"
            "% Unemployed: %{customdata[9]:.1f}%<br>"
            "% Renter: %{customdata[10]:.1f}%<br>"
            "Med HH Income: $%{customdata[11]:.1f}k<br>"
            "<extra></extra>"
        )
    else:
        st.warning("⚠️ No choropleth trace created — check the geometry match message above.")

    fig.update_geos(fitbounds="locations", visible=False)
    
    # Apply basemap style
    if basemap_style != "none":
        fig.update_geos(
            projection_type="mercator",
            showcountries=False,
            showcoastlines=False,
            showland=True,
            landcolor="lightgray",
            bgcolor="white"
        )
    
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))

    # Cornerstone markers on top with enhanced visibility
    if not sites_df.empty:
        fig.add_scattergeo(
            lat=sites_df["lat"],
            lon=sites_df["lon"],
            text=sites_df["name"],
            mode="markers+text",
            textposition="top center",
            textfont=dict(size=12, color="darkblue", family="Arial Black"),
            marker=dict(size=15, symbol="star", color="gold", line=dict(width=2, color="darkblue")),
            name="Cornerstone Sites",
            showlegend=True,
        )

    st.plotly_chart(fig, use_container_width=True)


def render_cards(df: pd.DataFrame) -> None:
    """Display highlight cards and top education-desert block groups table."""
    # Highlight cards with key statistics
    st.subheader("📊 Key Statistics - Philadelphia Block Groups")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Block Groups",
            value=f"{len(df):,}",
            help="Number of block groups analyzed"
        )
    
    with col2:
        avg_edi = df["edi_scaled"].mean()
        st.metric(
            label="Avg Education Desert Index",
            value=f"{avg_edi:.1f}",
            help="Average EDI score across all block groups"
        )
    
    with col3:
        high_need = (df["edi_scaled"] >= df["edi_scaled"].quantile(0.75)).sum()
        st.metric(
            label="High-Need Areas",
            value=f"{high_need:,}",
            help="Block groups in top 25% EDI (highest need)"
        )
    
    with col4:
        avg_poverty = df["pct_poverty"].mean()
        st.metric(
            label="Avg Poverty Rate",
            value=f"{avg_poverty:.1f}%",
            help="Average poverty rate across block groups"
        )
    
    st.markdown("---")
    
    st.subheader("🎯 Top Education-Desert Block Groups (Philadelphia)")
    top = df.sort_values("edi_scaled", ascending=False).head(10).copy()
    display = top[[
        "NAME", "edi_scaled", "pct_lt_hs", "pct_bach_plus",
        "pct_no_vehicle", "pct_no_inet", "pct_children", "pct_poverty", 
        "pct_unemployed", "pct_renter", "mhhinc_k"
    ]]
    display = display.rename(columns={
        "NAME": "Block Group",
        "edi_scaled": "EDI (0–100)",
        "pct_lt_hs": "% < HS",
        "pct_bach_plus": "% Bachelor's+",
        "pct_no_vehicle": "% HHs No Vehicle",
        "pct_no_inet": "% HHs No Internet",
        "pct_children": "% < 18",
        "pct_poverty": "% Poverty",
        "pct_unemployed": "% Unemployed",
        "pct_renter": "% Renter",
        "mhhinc_k": "Median HH Income ($k)",
    })
    display = display.round({
        "EDI (0–100)": 1, "% < HS": 1, "% Bachelor's+": 1,
        "% HHs No Vehicle": 1, "% HHs No Internet": 1, "% < 18": 1,
        "% Poverty": 1, "% Unemployed": 1, "% Renter": 1, "Median HH Income ($k)": 1
    })
    st.dataframe(display.fillna(""), use_container_width=True, height=400)


def render_download(df: pd.DataFrame) -> None:
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download block-group dataset (CSV)",
        data=csv_bytes,
        file_name="philly_education_desert_blockgroups.csv",
        mime="text/csv",
    )




def main() -> None:
    st.set_page_config(page_title="Philadelphia Education Desert (Block Groups)", layout="wide")
    st.title("📚 Philadelphia Education Desert Dashboard — Block-Group View")
    st.caption("ACS 5-year detailed tables at block-group level; composite index is relative within Philadelphia.")

    with st.sidebar:
        st.header("⚙️ Controls")
        
        # Year selector
        year = st.selectbox(
            "ACS 5-year vintage",
            options=AVAILABLE_YEARS,
            index=AVAILABLE_YEARS.index(ACS_YEAR_DEFAULT),
        )
        
        st.markdown("---")
        st.subheader("🗺️ Map Settings")
        
        # Metric selector
        metric_options = {
            "Education Desert Index": "edi_scaled",
            "% < HS": "pct_lt_hs",
            "% Bachelor's+": "pct_bach_plus",
            "% No Vehicle": "pct_no_vehicle",
            "% No Internet": "pct_no_inet",
            "% Children": "pct_children",
            "% Poverty": "pct_poverty",
            "% Unemployed": "pct_unemployed",
            "% Renter": "pct_renter",
            "Median HH Income ($k)": "mhhinc_k",
        }
        selected_metric_label = st.selectbox(
            "Map Metric",
            options=list(metric_options.keys()),
            index=0,
            help="Select which metric to display on the map"
        )
        selected_metric = metric_options[selected_metric_label]
        
        # Color scale selector
        color_scale = st.selectbox(
            "Color Scale",
            options=["YlOrRd", "RdYlGn_r", "Blues", "Viridis", "Plasma", "Inferno", "Turbo"],
            index=0,
            help="Color scheme for the map"
        )
        
        # Opacity slider
        opacity = st.slider(
            "Map Opacity",
            min_value=0.1,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Adjust transparency of map colors"
        )
        
        # Basemap style
        basemap_style = st.selectbox(
            "Basemap Style",
            options=["none", "carto-positron", "open-street-map", "white-bg"],
            index=0,
            help="Background map style"
        )
        
        st.markdown("---")
        st.subheader("🎯 Filters")
        
        # Tier filter
        tier = st.multiselect(
            "Show EDI Tiers", 
            ["Higher", "Moderate", "Lower"],
            default=["Higher", "Moderate", "Lower"],
            help="Filter block groups by Education Desert Index tier"
        )

    # Read key from secrets/env with clear messaging
    API_KEY = st.secrets.get("CENSUS_API_KEY", os.getenv("CENSUS_API_KEY", "")) or None

    # Helpful status chip with better messaging
    if API_KEY:
        st.success("✅ Census API key loaded from secrets.", icon="🔑")
    else:
        st.error(
            "❌ No Census API key found. Please add `CENSUS_API_KEY` to `.streamlit/secrets.toml` "
            "or set it as an environment variable. Get your free key at: https://api.census.gov/data/key_signup.html",
            icon="🔑"
        )
        st.stop()

    # Load data with spinner
    try:
        with st.spinner("Loading ACS data..."):
            df = load_acs_bg(year, API_KEY)
    except Exception as e:
        st.error(f"❌ Data load failed: {e}")
        st.info("💡 If you're seeing an API error, check your API key or try again later (Census API has rate limits).")
        return

    # Get site points
    sites = get_site_points(CCA_SITES)
    
    # Apply tier filter
    view = df[df["edi_tier"].astype(str).isin(tier)].copy() if tier else df.copy()

    # Tabbed interface
    tab1, tab2, tab3, tab4 = st.tabs(["📍 Map View", "🏆 Rankings", "📊 Detailed Data", "⬇️ Download"])
    
    with tab1:
        st.subheader(f"Map: {selected_metric_label}")
        render_map(view, sites, metric=selected_metric, color_scale=color_scale, 
                   opacity=opacity, basemap_style=basemap_style)
    
    with tab2:
        render_cards(view)
    
    with tab3:
        st.subheader("📊 Complete Block Group Data")
        st.caption(f"Showing {len(view)} of {len(df)} block groups (filtered by tier selection)")
        
        # Full data table with all metrics
        detail_cols = [
            "NAME", "geoid_bg", "edi_scaled", "edi_tier",
            "pct_lt_hs", "pct_bach_plus", "pct_no_vehicle", "pct_no_inet", 
            "pct_children", "pct_poverty", "pct_unemployed", "pct_renter", "mhhinc_k"
        ]
        detail_df = view[detail_cols].copy()
        detail_df = detail_df.rename(columns={
            "NAME": "Block Group",
            "geoid_bg": "GEOID",
            "edi_scaled": "EDI (0–100)",
            "edi_tier": "EDI Tier",
            "pct_lt_hs": "% < HS",
            "pct_bach_plus": "% Bachelor's+",
            "pct_no_vehicle": "% No Vehicle",
            "pct_no_inet": "% No Internet",
            "pct_children": "% < 18",
            "pct_poverty": "% Poverty",
            "pct_unemployed": "% Unemployed",
            "pct_renter": "% Renter",
            "mhhinc_k": "Med HH Inc ($k)",
        })
        
        # Sort by EDI descending
        detail_df = detail_df.sort_values("EDI (0–100)", ascending=False)
        
        st.dataframe(detail_df.fillna(""), use_container_width=True, height=600)
    
    with tab4:
        st.subheader("⬇️ Download Data")
        st.write("Download the complete block-group dataset with all calculated metrics.")
        render_download(view)
        
        st.markdown("---")
        st.info(
            "**Data includes:** Block group identifiers, Education Desert Index scores, "
            "educational attainment, access metrics (vehicle/internet), demographics "
            "(children, poverty, unemployment, renters), and median household income."
        )

    # Methodology section
    with st.expander("📖 Methodology & Data Sources"):
        st.markdown(
            """
### Methodology (Block-Group Level)

**Education Desert Index (EDI)** combines three pillars:

1. **Need Score**  
   - % adults 25+ with < HS diploma (B15003 cells 2–16 / 001)  
   - % population under 18 (B09001_001E / B01003_001E)

2. **Choice Gap Score**  
   - % < HS offset by low % Bachelor's+ (B15003 cells 21–24 / 001)

3. **Access Friction Score**  
   - % households without a vehicle (B08201_002E / 001E)  
   - % households without Internet (B28002_013E / 001E)  
   - Median HH income (B19013_001E, inverted for scoring)

**Additional Metrics:**
- **Poverty**: % below poverty line (B17001_002E / B17001_001E)
- **Unemployment**: % unemployed in labor force (B23025_005E / B23025_001E)  
- **Renters**: % renter-occupied housing units (B25003_003E / B25003_001E)

**Composite**: All pillars are z-scored, averaged, and scaled to 0–100. Rankings are relative within **Philadelphia County** only.

### Data Sources
- **ACS 5-Year Estimates**: US Census Bureau American Community Survey
- **Geography**: TIGERweb Block Group boundaries for Philadelphia County (FIPS 42101)
- **Cornerstone Sites**: Geocoded via Census Geocoding API
"""
        )

    # Footer
    st.markdown("---")
    st.caption(
        "Built with Streamlit • Data from US Census Bureau ACS 5-Year Estimates • "
        f"Viewing {year} vintage"
    )


if __name__ == "__main__":
    main()
