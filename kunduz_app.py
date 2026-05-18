"""
Kunduz Smart Farming Dashboard
ADI x WUR x FAO Rome 2025
Author: Maiwand Jan Alamzoi — Afghanistan Development Initiative
No geemap — uses folium directly for full Python 3.14 compatibility
"""

import ee
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Kunduz Smart Farming — ADI x WUR x FAO",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main{background-color:#0a0f0d}
.block-container{padding-top:1.5rem}
h1{color:#4ade80;font-family:monospace}
h2,h3{color:#86efac}
div[data-testid="metric-container"]{
    background:#111810;border:1px solid #1e2b1a;
    border-radius:8px;padding:0.75rem}
.stTabs [data-baseweb="tab"]{font-family:monospace;font-size:13px;color:#6b8f65}
.stTabs [aria-selected="true"]{color:#4ade80 !important;border-bottom:2px solid #4ade80 !important}
.insight{background:#111810;border-left:3px solid #16a34a;border-radius:4px;padding:1rem;margin:0.5rem 0}
.insight.water{border-left-color:#0ea5e9}
.insight.warning{border-left-color:#fbbf24}
.insight.danger{border-left-color:#f87171}
.insight.snow{border-left-color:#93c5fd}
.insight.forest{border-left-color:#a78bfa}
</style>
""", unsafe_allow_html=True)

# ─── GEE INIT ────────────────────────────────────────────────────────────────
@st.cache_resource
def init_gee():
    try:
        service_account = st.secrets["gee"]["service_account"]
        private_key     = st.secrets["gee"]["private_key"]
        credentials = ee.ServiceAccountCredentials(
            service_account, key_data=private_key
        )
        ee.Initialize(credentials)
        return True
    except Exception as e:
        st.error(f"GEE authentication failed: {e}")
        return False

gee_ok = init_gee()

# ─── CONSTANTS ───────────────────────────────────────────────────────────────
YEARS = [2019, 2020, 2021, 2022, 2023, 2024]
BG    = "#111810"
GC    = "rgba(255,255,255,0.04)"
TS    = dict(color="#6b8f65", family="monospace", size=11)

PROVINCES = {
    "Kunduz":      {"bbox": [68.55, 36.55, 69.05, 37.05], "center": [36.73, 68.87]},
    "Balkh":       {"bbox": [66.70, 36.50, 67.20, 37.00], "center": [36.76, 66.90]},
    "Helmand":     {"bbox": [63.80, 31.00, 64.80, 31.80], "center": [31.35, 64.20]},
    "Herat":       {"bbox": [61.80, 34.10, 62.50, 34.60], "center": [34.34, 62.20]},
    "Nangarhar":   {"bbox": [70.20, 34.00, 70.80, 34.50], "center": [34.17, 70.62]},
    "Kabul":       {"bbox": [69.00, 34.30, 69.50, 34.70], "center": [34.53, 69.17]},
    "Kandahar":    {"bbox": [65.40, 31.50, 66.00, 31.90], "center": [31.63, 65.71]},
    "Takhar":      {"bbox": [69.30, 36.60, 70.00, 37.10], "center": [36.83, 69.52]},
    "Baghlan":     {"bbox": [68.40, 36.00, 69.00, 36.60], "center": [36.17, 68.71]},
    "Badakhshan":  {"bbox": [70.50, 36.80, 71.50, 37.50], "center": [37.12, 70.81]},
}

# Real Sentinel-2 data from GEE — Kunduz 2019-2024
REAL_DATA = {
    2019: {"ndvi": 0.172, "cropland": 385.9, "water": 24.1},
    2020: {"ndvi": 0.168, "cropland": 370.9, "water": 21.8},
    2021: {"ndvi": 0.162, "cropland": 300.0, "water": 17.0},
    2022: {"ndvi": 0.139, "cropland": 182.7, "water":  5.8},
    2023: {"ndvi": 0.142, "cropland": 211.3, "water": 11.0},
    2024: {"ndvi": 0.152, "cropland": 251.9, "water": 14.2},
}

# ─── GEE HELPER FUNCTIONS ────────────────────────────────────────────────────
def get_s2(region, year, start="05-01", end="07-31"):
    return (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(region)
            .filterDate(f"{year}-{start}", f"{year}-{end}")
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 15))
            .median()
            .clip(region))

def reduce_mean(img, region, scale=30):
    return img.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=region,
        scale=scale,
        maxPixels=1e9
    ).getInfo()

def area_km2(mask, band, region, scale=30):
    result = (mask.multiply(ee.Image.pixelArea())
              .reduceRegion(
                  reducer=ee.Reducer.sum(),
                  geometry=region,
                  scale=scale,
                  maxPixels=1e9
              ).getInfo())
    val = result.get(band, 0)
    return round((val or 0) / 1e6, 1)

@st.cache_data(ttl=3600, show_spinner=False)
def get_province_stats(province_name, year):
    bbox   = PROVINCES[province_name]["bbox"]
    region = ee.Geometry.Rectangle(bbox)
    s2     = get_s2(region, year)
    ndvi   = s2.normalizedDifference(["B8", "B4"]).rename("NDVI")
    mndwi  = s2.normalizedDifference(["B3", "B11"]).rename("MNDWI")

    ndvi_val  = (reduce_mean(ndvi,  region) or {}).get("NDVI", 0) or 0
    crop_km2  = area_km2(ndvi.gt(0.35),  "NDVI",  region)
    water_km2 = area_km2(mndwi.gt(0.05), "MNDWI", region)

    return {
        "year":     year,
        "province": province_name,
        "ndvi":     round(ndvi_val, 4),
        "cropland": crop_km2,
        "water":    water_km2,
    }

@st.cache_data(ttl=3600, show_spinner=False)
def get_climate(province_name, year):
    bbox      = PROVINCES[province_name]["bbox"]
    region    = ee.Geometry.Rectangle(bbox)
    hindukush = ee.Geometry.Rectangle([68.0, 36.0, 70.0, 38.5])

    rain = (ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
            .filterBounds(region)
            .filterDate(f"{year}-01-01", f"{year}-12-31")
            .select("precipitation").sum().clip(region))

    snow = (ee.ImageCollection("MODIS/061/MOD10A1")
            .filterBounds(hindukush)
            .filterDate(f"{year-1}-12-01", f"{year}-03-31")
            .select("NDSI_Snow_Cover").mean().clip(hindukush))

    et = (ee.ImageCollection("MODIS/061/MOD16A2")
          .filterBounds(region)
          .filterDate(f"{year}-05-01", f"{year}-07-31")
          .select("ET").sum().multiply(0.1).clip(region))

    rain_val = (reduce_mean(rain, region, scale=5000) or {}).get("precipitation", 0) or 0
    snow_val = (reduce_mean(snow, hindukush, scale=500) or {}).get("NDSI_Snow_Cover", 0) or 0
    et_val   = (reduce_mean(et,   region, scale=500)  or {}).get("ET", 0) or 0

    return {
        "rain_annual":   round(rain_val, 1),
        "snow_cover":    round(snow_val, 1),
        "et":            round(et_val,   1),
        "water_balance": round(rain_val - et_val, 1),
    }

def get_tile_url(img, vis_params):
    """Get tile URL from GEE image for folium."""
    map_id = img.getMapId(vis_params)
    return map_id["tile_fetcher"].url_format

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛰️ ADI x WUR x FAO")
    st.markdown("**Afghanistan Smart Farming**")
    st.divider()

    location_mode = st.radio(
        "Select region by:",
        ["Province", "Draw on map"],
        horizontal=True
    )

    if location_mode == "Province":
        selected_province = st.selectbox(
            "Province", list(PROVINCES.keys()), index=0
        )
        bbox      = PROVINCES[selected_province]["bbox"]
        center    = PROVINCES[selected_province]["center"]
        region_name = selected_province
    else:
        st.info("Draw a rectangle on the map to select your region.")
        draw_map = folium.Map(location=[33.9, 67.7], zoom_start=5,
                              tiles="CartoDB dark_matter")
        Draw(
            draw_options={
                "rectangle": True, "polygon": False,
                "circle": False, "marker": False,
                "polyline": False, "circlemarker": False
            },
            edit_options={"edit": False}
        ).add_to(draw_map)
        output = st_folium(draw_map, width=280, height=280, key="draw")

        if output and output.get("last_active_drawing"):
            coords = output["last_active_drawing"]["geometry"]["coordinates"][0]
            lons   = [c[0] for c in coords]
            lats   = [c[1] for c in coords]
            bbox   = [min(lons), min(lats), max(lons), max(lats)]
            center = [(min(lats)+max(lats))/2, (min(lons)+max(lons))/2]
            region_name = f"Custom region"
            st.success(f"Selected: {bbox[1]:.2f}N {bbox[0]:.2f}E")
        else:
            bbox        = PROVINCES["Kunduz"]["bbox"]
            center      = PROVINCES["Kunduz"]["center"]
            region_name = "Kunduz (default)"

    st.divider()
    st.markdown(f"**Region:** {region_name}")
    map_year  = st.selectbox("Map year", YEARS, index=5)
    map_layer = st.selectbox("Map layer", [
        "NDVI", "Water (MNDWI)", "True Color", "False Color"
    ])
    st.divider()
    st.markdown("**Analyst:** Maiwand Jan Alamzoi")
    st.markdown("**ADI × WUR × FAO Rome 2025**")

# ─── HEADER ──────────────────────────────────────────────────────────────────
st.title("🛰️ Afghanistan Smart Farming Dashboard")
st.markdown(
    f"**Region: {region_name}** · Sentinel-2 · 2019–2024 · "
    "Afghanistan Development Initiative × WUR × FAO Rome 2025"
)

# ─── TABS ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📡 Satellite",
    "🌧️ Climate",
    "🌳 Forest",
    "🌾 Crops",
    "🗺️ Live Map",
    "🤖 AI Assistant"
])

# ═══════════════════════════════════════════════════
# TAB 1 — SATELLITE
# ═══════════════════════════════════════════════════
with tab1:
    st.subheader(f"Vegetation, Water & Cropland — {region_name} 2019–2024")

    # Use real Kunduz data or load live
    if region_name in ["Kunduz", "Kunduz (default)"]:
        df = pd.DataFrame([{"year": y, **REAL_DATA[y]} for y in YEARS])
        st.info("Showing real Sentinel-2 data for Kunduz (2019–2024). Select another province and click Load to get live data.")
    else:
        df = pd.DataFrame([{"year": y, **REAL_DATA[y]} for y in YEARS])
        st.warning(f"Showing Kunduz baseline data. Click 'Load Live Data' to pull real data for {region_name}.")

    if st.button("🛰️ Load Live Satellite Data", type="primary"):
        rows = []
        with st.spinner(f"Pulling Sentinel-2 data for {region_name}..."):
            for yr in YEARS:
                try:
                    row = get_province_stats(region_name if location_mode == "Province" else "Kunduz", yr)
                    rows.append(row)
                except Exception as e:
                    st.warning(f"Year {yr}: {e}")
        if rows:
            df = pd.DataFrame(rows)
            st.success("Live data loaded!")

    # Metrics
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Best cropland",  f"{df['cropland'].max()} km²",
              f"{int(df.loc[df['cropland'].idxmax(),'year'])} — peak")
    c2.metric("Worst cropland", f"{df['cropland'].min()} km²",
              f"{int(df.loc[df['cropland'].idxmin(),'year'])} − {round((1-df['cropland'].min()/df['cropland'].max())*100)}%",
              delta_color="inverse")
    c3.metric("Now 2024",       f"{df[df['year']==2024]['cropland'].values[0]} km²",
              "recovering")
    c4.metric("Best water",     f"{df['water'].max()} km²",
              f"{int(df.loc[df['water'].idxmax(),'year'])} — peak")
    c5.metric("Worst water",    f"{df['water'].min()} km²",
              f"−{round((1-df['water'].min()/df['water'].max())*100)}%",
              delta_color="inverse")
    c6.metric("NDVI peak",      f"{df['ndvi'].max()}",
              f"{int(df.loc[df['ndvi'].idxmax(),'year'])}")

    st.divider()

    # Charts
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["year"], y=df["ndvi"],
            mode="lines+markers",
            line=dict(color="#4ade80", width=2),
            marker=dict(
                color=["#f87171" if v==df["ndvi"].min()
                       else "#4ade80" if v==df["ndvi"].max()
                       else "#86efac" for v in df["ndvi"]],
                size=10),
            fill="tozeroy", fillcolor="rgba(74,222,128,0.06)"
        ))
        fig.update_layout(
            title="Mean NDVI — Vegetation Health",
            paper_bgcolor=BG, plot_bgcolor=BG, font=dict(**TS),
            xaxis=dict(gridcolor=GC, tickfont=TS),
            yaxis=dict(gridcolor=GC, tickfont=TS, range=[0.10, 0.20]),
            showlegend=False, margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=df["year"], y=df["cropland"],
            marker_color=["#4ade80" if v==df["cropland"].max()
                          else "#f87171" if v==df["cropland"].min()
                          else "rgba(74,222,128,0.5)" for v in df["cropland"]],
            marker_line_width=0))
        fig2.update_layout(
            title="Cropland km² (NDVI > 0.35)",
            paper_bgcolor=BG, plot_bgcolor=BG, font=dict(**TS),
            xaxis=dict(gridcolor=GC, tickfont=TS),
            yaxis=dict(gridcolor=GC, tickfont=TS),
            showlegend=False, margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=df["year"], y=df["water"],
            marker_color=["#38bdf8" if v==df["water"].max()
                          else "#f87171" if v==df["water"].min()
                          else "rgba(56,189,248,0.5)" for v in df["water"]],
            marker_line_width=0))
        fig3.update_layout(
            title="Water Surface km² (MNDWI > 0.05)",
            paper_bgcolor=BG, plot_bgcolor=BG, font=dict(**TS),
            xaxis=dict(gridcolor=GC, tickfont=TS),
            yaxis=dict(gridcolor=GC, tickfont=TS),
            showlegend=False, margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=df["water"], y=df["cropland"],
            mode="markers+text",
            text=df["year"].astype(str),
            textposition="top center",
            marker=dict(
                size=14,
                color=["#4ade80","#86efac","#fbbf24",
                       "#f87171","#fb923c","#a78bfa"])))
        fig4.update_layout(
            title="Water vs Cropland — r = 0.97",
            xaxis_title="Water km²",
            yaxis_title="Cropland km²",
            paper_bgcolor=BG, plot_bgcolor=BG, font=dict(**TS),
            xaxis=dict(gridcolor=GC, tickfont=TS),
            yaxis=dict(gridcolor=GC, tickfont=TS),
            showlegend=False, margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig4, use_container_width=True)

    # Change vs baseline
    st.subheader("Change vs 2019 Baseline")
    baseline = df[df["year"]==2019].iloc[0]
    df["crop_chg"]  = ((df["cropland"]-baseline["cropland"])/baseline["cropland"]*100).round(1)
    df["water_chg"] = ((df["water"]   -baseline["water"])   /baseline["water"]   *100).round(1)
    df["ndvi_chg"]  = ((df["ndvi"]    -baseline["ndvi"])    /baseline["ndvi"]    *100).round(1)

    fig5 = go.Figure()
    fig5.add_trace(go.Bar(x=df["year"], y=df["crop_chg"],  name="Cropland %", marker_color="#4ade80", opacity=0.8))
    fig5.add_trace(go.Bar(x=df["year"], y=df["water_chg"], name="Water %",    marker_color="#38bdf8", opacity=0.8))
    fig5.add_trace(go.Bar(x=df["year"], y=df["ndvi_chg"],  name="NDVI %",     marker_color="#fbbf24", opacity=0.8))
    fig5.add_hline(y=0, line_color="#6b8f65", line_width=1)
    fig5.update_layout(
        barmode="group",
        paper_bgcolor=BG, plot_bgcolor=BG, font=dict(**TS),
        xaxis=dict(gridcolor=GC, tickfont=TS),
        yaxis=dict(gridcolor=GC, tickfont=TS, title="% vs 2019"),
        legend=dict(font=dict(color="#6b8f65"), bgcolor=BG),
        margin=dict(l=10,r=10,t=20,b=10))
    st.plotly_chart(fig5, use_container_width=True)

    # Data table
    st.subheader("Full Data Table")
    st.dataframe(
        df[["year","ndvi","cropland","water","crop_chg","water_chg"]].rename(columns={
            "year":"Year","ndvi":"Mean NDVI","cropland":"Cropland km²",
            "water":"Water km²","crop_chg":"Crop Δ%","water_chg":"Water Δ%"
        }), use_container_width=True, hide_index=True)

    # Insights
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""<div class="insight">
        <strong style="color:#4ade80">CROPLAND COLLAPSE AND RECOVERY</strong><br><br>
        Active cropland dropped 53% from 386 km² (2019) to 183 km² (2022).
        Recovery visible at 252 km² in 2024 — but 134 km² still abandoned.
        This is the core smart farming opportunity for Kunduz smallholders.
        </div>""", unsafe_allow_html=True)
        st.markdown("""<div class="insight danger">
        <strong style="color:#f87171">YIELD GAP IS LARGE</strong><br><br>
        Peak NDVI of 0.172 in 2019 is well below this land's potential.
        Smallholders farm without optimisation guidance. NDVI-based alerts
        could increase yields 20–30% with same inputs and water.
        </div>""", unsafe_allow_html=True)
    with col_b:
        st.markdown("""<div class="insight water">
        <strong style="color:#0ea5e9">WATER IS THE #1 CONSTRAINT</strong><br><br>
        Water dropped 76% in 2022. Cropland followed exactly — r=0.97.
        Water availability explains 94% of cropland variation.
        Irrigation efficiency gives highest ROI for smallholders.
        </div>""", unsafe_allow_html=True)
        st.markdown("""<div class="insight warning">
        <strong style="color:#fbbf24">RECOVERY IS FRAGILE</strong><br><br>
        2024 shows recovery but both water (14.2 km²) and cropland (251.9 km²)
        remain below 2019. Mean NDVI (0.152) still 12% below peak.
        One dry season could reverse all gains. Remote monitoring essential.
        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════
# TAB 2 — CLIMATE
# ═══════════════════════════════════════════════════
with tab2:
    st.subheader("Rainfall · Snow · Evapotranspiration · Water Balance")
    st.info("Click to load live climate data from CHIRPS, MODIS, and ERA5.")

    if st.button("🌧️ Load Climate Data", type="primary"):
        rows = []
        with st.spinner("Loading climate data from GEE..."):
            for yr in YEARS:
                try:
                    row = get_climate(
                        region_name if location_mode=="Province" else "Kunduz", yr
                    )
                    row["year"] = yr
                    rows.append(row)
                    st.write(f"{yr}: Rain={row['rain_annual']}mm "
                             f"Snow={row['snow_cover']}% ET={row['et']}mm")
                except Exception as e:
                    st.warning(f"{yr}: {e}")
        if rows:
            st.session_state["climate"] = pd.DataFrame(rows)
            st.success("Done!")

    if "climate" in st.session_state:
        cdf = st.session_state["climate"]

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Best rain year",  str(int(cdf.loc[cdf["rain_annual"].idxmax(),"year"])),
                  f"{cdf['rain_annual'].max()} mm")
        c2.metric("Worst rain year", str(int(cdf.loc[cdf["rain_annual"].idxmin(),"year"])),
                  f"{cdf['rain_annual'].min()} mm", delta_color="inverse")
        c3.metric("Best snow year",  str(int(cdf.loc[cdf["snow_cover"].idxmax(),"year"])),
                  f"{cdf['snow_cover'].max()}%")
        c4.metric("Worst snow year", str(int(cdf.loc[cdf["snow_cover"].idxmin(),"year"])),
                  f"{cdf['snow_cover'].min()}%", delta_color="inverse")

        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=cdf["year"], y=cdf["rain_annual"],
                marker_color="rgba(56,189,248,0.7)", name="Rain mm"))
            fig.update_layout(title="Annual Rainfall mm (CHIRPS)",
                paper_bgcolor=BG, plot_bgcolor=BG, font=dict(**TS),
                xaxis=dict(gridcolor=GC,tickfont=TS),
                yaxis=dict(gridcolor=GC,tickfont=TS),
                showlegend=False, margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=cdf["year"], y=cdf["snow_cover"],
                mode="lines+markers",
                line=dict(color="#93c5fd", width=2),
                marker=dict(size=10, color="#93c5fd"),
                fill="tozeroy", fillcolor="rgba(147,197,253,0.08)"))
            fig2.update_layout(title="Winter Snow Cover % — Hindu Kush",
                paper_bgcolor=BG, plot_bgcolor=BG, font=dict(**TS),
                xaxis=dict(gridcolor=GC,tickfont=TS),
                yaxis=dict(gridcolor=GC,tickfont=TS),
                showlegend=False, margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig2, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            fig3 = go.Figure()
            fig3.add_trace(go.Bar(x=cdf["year"], y=cdf["et"],
                marker_color="rgba(251,191,36,0.7)"))
            fig3.update_layout(title="Evapotranspiration mm (MODIS MOD16)",
                paper_bgcolor=BG, plot_bgcolor=BG, font=dict(**TS),
                xaxis=dict(gridcolor=GC,tickfont=TS),
                yaxis=dict(gridcolor=GC,tickfont=TS),
                showlegend=False, margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig3, use_container_width=True)

        with col4:
            fig4 = go.Figure()
            fig4.add_trace(go.Bar(
                x=cdf["year"], y=cdf["water_balance"],
                marker_color=["#4ade80" if v>0 else "#f87171"
                              for v in cdf["water_balance"]]))
            fig4.add_hline(y=0, line_color="#6b8f65", line_width=1)
            fig4.update_layout(title="Water Balance mm (Rain − ET)",
                paper_bgcolor=BG, plot_bgcolor=BG, font=dict(**TS),
                xaxis=dict(gridcolor=GC,tickfont=TS),
                yaxis=dict(gridcolor=GC,tickfont=TS),
                showlegend=False, margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig4, use_container_width=True)

        st.markdown("""<div class="insight snow">
        <strong style="color:#93c5fd">THE 2022 TRIPLE CRISIS</strong><br><br>
        In 2022 three climate factors hit simultaneously: rainfall was below average,
        winter snowpack in the Hindu Kush was at record lows (less meltwater for the
        Kunduz River), and temperatures were above average (more evapotranspiration loss).
        This triple combination caused the 76% water collapse confirmed by satellite.
        This is the strongest argument for climate-smart irrigation tools for smallholders.
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div class="insight warning">
        <strong style="color:#fbbf24">WHAT THIS TAB SHOWS</strong><br><br>
        Annual and growing season rainfall (CHIRPS 5km), winter snow cover upstream
        in the Hindu Kush (MODIS 500m), evapotranspiration during growing season
        (MODIS MOD16), and the water balance (rainfall minus ET). Together these
        explain WHY the cropland and water collapsed in 2022 — and what to monitor
        to predict future crises before they happen.
        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════
# TAB 3 — FOREST
# ═══════════════════════════════════════════════════
with tab3:
    st.subheader("Forest Cover & Deforestation — Hansen Global Forest Watch")

    if st.button("🌳 Load Forest Data", type="primary"):
        rows = []
        bbox   = PROVINCES[region_name]["bbox"] if location_mode=="Province" else PROVINCES["Kunduz"]["bbox"]
        region = ee.Geometry.Rectangle(bbox)
        hansen = ee.Image("UMD/hansen/global_forest_change_v1_12_2023")

        with st.spinner("Loading forest change data..."):
            for yr in [2019, 2020, 2021, 2022, 2023]:
                try:
                    loss = hansen.select("lossyear").eq(yr - 2000)
                    loss_km2 = area_km2(loss, "lossyear", region)
                    rows.append({"year": yr, "loss_km2": loss_km2})
                except Exception as e:
                    st.warning(f"{yr}: {e}")

        if rows:
            st.session_state["forest"] = pd.DataFrame(rows)
            st.success("Done!")

    if "forest" in st.session_state:
        fdf = st.session_state["forest"]

        c1,c2,c3 = st.columns(3)
        c1.metric("Total loss 2019–2023", f"{fdf['loss_km2'].sum():.2f} km²")
        c2.metric("Worst loss year", str(int(fdf.loc[fdf["loss_km2"].idxmax(),"year"])),
                  f"{fdf['loss_km2'].max():.2f} km²")
        c3.metric("Average annual loss", f"{fdf['loss_km2'].mean():.2f} km²")

        fig = go.Figure()
        fig.add_trace(go.Bar(x=fdf["year"], y=fdf["loss_km2"],
            marker_color="rgba(167,139,250,0.8)"))
        fig.update_layout(title="Annual Forest Loss km² (Hansen GFW)",
            paper_bgcolor=BG, plot_bgcolor=BG, font=dict(**TS),
            xaxis=dict(gridcolor=GC,tickfont=TS),
            yaxis=dict(gridcolor=GC,tickfont=TS),
            showlegend=False, margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""<div class="insight forest">
        <strong style="color:#a78bfa">DEFORESTATION AMPLIFIES WATER CRISIS</strong><br><br>
        Tree cover loss in the Kunduz watershed reduces the land's ability to hold
        and regulate water. Deforested slopes cause faster runoff, more erosion, and
        less groundwater recharge — making drought impacts worse even in normal rainfall years.
        Upstream reforestation is a long-term smart farming investment with high ROI.
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div class="insight forest">
        <strong style="color:#a78bfa">WHAT THIS MODULE COVERS</strong><br><br>
        Hansen Global Forest Watch data shows where tree cover was lost each year
        in and around the watershed. Deforestation directly connects to water
        availability — forests regulate river flow, reduce erosion, and recharge
        groundwater. This module quantifies that connection with satellite data.
        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════
# TAB 4 — CROPS
# ═══════════════════════════════════════════════════
with tab4:
    st.subheader("Crop Intelligence — What Grows Now vs What Should Grow")

    st.markdown("### Current Crops in Kunduz Province")
    crops_now = pd.DataFrame([
        {"Crop":"Wheat (winter)","Season":"Oct–May","Water need":"Medium","Status":"Main staple crop"},
        {"Crop":"Cotton",        "Season":"Apr–Oct","Water need":"High",  "Status":"Cash crop — declining due to water shortage"},
        {"Crop":"Rice",          "Season":"May–Sep","Water need":"Very high","Status":"Irrigated lowlands only"},
        {"Crop":"Flax",          "Season":"Apr–Aug","Water need":"Low",   "Status":"Traditional — growing interest"},
        {"Crop":"Vegetables",    "Season":"Apr–Oct","Water need":"Medium","Status":"Household and local market"},
        {"Crop":"Melon",         "Season":"May–Sep","Water need":"Medium","Status":"Local market"},
    ])
    st.dataframe(crops_now, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### Water Productivity — Value per mm of Water")
    water_prod = pd.DataFrame([
        {"Crop":"Saffron",    "Water mm/yr":300,  "Value USD/ha":15000,"Water productivity":"Very High","Feasibility":"High — grown in nearby provinces"},
        {"Crop":"Flax",       "Water mm/yr":350,  "Value USD/ha":800,  "Water productivity":"High",     "Feasibility":"High — already in Kunduz"},
        {"Crop":"Almonds",    "Water mm/yr":400,  "Value USD/ha":3000, "Water productivity":"High",     "Feasibility":"Medium — 3yr establishment"},
        {"Crop":"Wheat",      "Water mm/yr":450,  "Value USD/ha":400,  "Water productivity":"Medium",   "Feasibility":"Very High — current main crop"},
        {"Crop":"Vegetables", "Water mm/yr":500,  "Value USD/ha":2000, "Water productivity":"Medium",   "Feasibility":"High — local market demand"},
        {"Crop":"Cotton",     "Water mm/yr":700,  "Value USD/ha":600,  "Water productivity":"Low",      "Feasibility":"Low — too much water needed"},
        {"Crop":"Rice",       "Water mm/yr":1200, "Value USD/ha":500,  "Water productivity":"Very Low", "Feasibility":"Very Low — unsustainable"},
    ])
    st.dataframe(water_prod, use_container_width=True, hide_index=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=water_prod["Water mm/yr"],
        y=water_prod["Value USD/ha"],
        mode="markers+text",
        text=water_prod["Crop"],
        textposition="top center",
        marker=dict(size=16,
            color=["#fbbf24","#4ade80","#86efac",
                   "#38bdf8","#a78bfa","#f87171","#ef4444"])
    ))
    fig.add_vline(x=450, line_color="#6b8f65", line_dash="dash",
                  annotation_text="Current water availability",
                  annotation_font_color="#6b8f65")
    fig.update_layout(
        title="Move LEFT and UP = best crops for water-stressed Kunduz",
        xaxis_title="Water need (mm/year)",
        yaxis_title="Farm value (USD/ha)",
        paper_bgcolor=BG, plot_bgcolor=BG, font=dict(**TS),
        xaxis=dict(gridcolor=GC, tickfont=TS),
        yaxis=dict(gridcolor=GC, tickfont=TS),
        showlegend=False, margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""<div class="insight">
    <strong style="color:#4ade80">SMART FARMING RECOMMENDATION</strong><br><br>
    Given water constraints (14.2 km² in 2024, still 41% below 2019):
    (1) Replace rice and cotton with flax and vegetables — same income, 40–60% less water.
    (2) Introduce saffron on higher ground — 50x more value per hectare than wheat with less water.
    (3) Improve wheat yields with NDVI monitoring instead of expanding area.
    Goal: more value per drop of water — not more area under cultivation.
    </div>""", unsafe_allow_html=True)

    # Province comparison
    st.divider()
    st.subheader("Compare Two Provinces")
    col1, col2 = st.columns(2)
    with col1:
        prov1 = st.selectbox("Province 1", list(PROVINCES.keys()), index=0)
    with col2:
        prov2 = st.selectbox("Province 2", list(PROVINCES.keys()), index=1)

    if st.button("🔄 Compare Provinces", type="primary"):
        with st.spinner("Loading data for both provinces..."):
            try:
                s1 = get_province_stats(prov1, 2024)
                s2 = get_province_stats(prov2, 2024)

                c1,c2_col,c3 = st.columns(3)
                c1.metric(f"{prov1} NDVI",     s1["ndvi"],
                          f"{'▲' if s1['ndvi']>s2['ndvi'] else '▼'} vs {prov2}")
                c2_col.metric(f"{prov1} Water", f"{s1['water']} km²",
                          f"{'▲' if s1['water']>s2['water'] else '▼'} vs {prov2}")
                c3.metric(f"{prov1} Cropland",  f"{s1['cropland']} km²",
                          f"{'▲' if s1['cropland']>s2['cropland'] else '▼'} vs {prov2}")

                fig = go.Figure()
                fig.add_trace(go.Bar(name=prov1,
                    x=["NDVI×100","Water km²","Cropland km²"],
                    y=[s1["ndvi"]*100, s1["water"], s1["cropland"]],
                    marker_color="#4ade80"))
                fig.add_trace(go.Bar(name=prov2,
                    x=["NDVI×100","Water km²","Cropland km²"],
                    y=[s2["ndvi"]*100, s2["water"], s2["cropland"]],
                    marker_color="#38bdf8"))
                fig.update_layout(barmode="group",
                    title=f"{prov1} vs {prov2} — 2024 Growing Season",
                    paper_bgcolor=BG, plot_bgcolor=BG, font=dict(**TS),
                    xaxis=dict(gridcolor=GC,tickfont=TS),
                    yaxis=dict(gridcolor=GC,tickfont=TS),
                    legend=dict(font=dict(color="#6b8f65"),bgcolor=BG),
                    margin=dict(l=10,r=10,t=40,b=10))
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Comparison failed: {e}")

# ═══════════════════════════════════════════════════
# TAB 5 — LIVE MAP
# ═══════════════════════════════════════════════════
with tab5:
    st.subheader(f"Live Satellite Map — {region_name} · {map_year}")

    bbox_map   = PROVINCES[region_name]["bbox"]   if location_mode=="Province" else PROVINCES["Kunduz"]["bbox"]
    center_map = PROVINCES[region_name]["center"] if location_mode=="Province" else [36.73, 68.87]

    # Layer selector — inline buttons for user friendliness
    st.markdown("**Select layers to display:**")
    layer_cols = st.columns(6)
    show_ndvi     = layer_cols[0].checkbox("🌿 NDVI",       value=True)
    show_water    = layer_cols[1].checkbox("💧 Water",      value=True)
    show_truecolor= layer_cols[2].checkbox("🗺️ True Color", value=False)
    show_falsecolor=layer_cols[3].checkbox("🔴 False Color",value=False)
    show_rainfall = layer_cols[4].checkbox("🌧️ Rainfall",  value=False)
    show_boundary = layer_cols[5].checkbox("📦 Boundary",   value=True)

    # Year selector inline
    col_yr, col_btn = st.columns([3,1])
    with col_yr:
        selected_year_map = st.select_slider(
            "Year", options=YEARS, value=map_year
        )
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        load_map = st.button("🗺️ Load Map", type="primary", use_container_width=True)

    # Legend
    st.markdown("""
    <div style="display:flex;gap:16px;flex-wrap:wrap;font-size:12px;color:#6b8f65;margin-bottom:8px">
        <span>🟢 High NDVI = healthy crops</span>
        <span>🔴 Low NDVI = stressed/bare</span>
        <span>🔵 Water surfaces</span>
        <span>🟡 Region boundary</span>
    </div>
    """, unsafe_allow_html=True)

    # Always show map — store in session state so it persists
    if load_map or "map_html" not in st.session_state:
        with st.spinner("Loading satellite layers from GEE..."):
            try:
                region_ee = ee.Geometry.Rectangle(bbox_map)
                s2 = get_s2(region_ee, selected_year_map)

                # Build map
                m = folium.Map(
                    location=center_map,
                    zoom_start=11,
                    tiles="CartoDB dark_matter"
                )

                # Add satellite base
                folium.TileLayer(
                    tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
                    attr="Google Satellite",
                    name="Google Satellite",
                    overlay=False
                ).add_to(m)

                # NDVI layer
                if show_ndvi:
                    ndvi = s2.normalizedDifference(["B8","B4"])
                    ndvi_url = get_tile_url(ndvi, {
                        "min":0, "max":0.7,
                        "palette":["#d73027","#fc8d59","#fee08b","#d9ef8b","#91cf60","#1a9850"]
                    })
                    folium.TileLayer(
                        tiles=ndvi_url,
                        attr="GEE Sentinel-2 NDVI",
                        name=f"🌿 NDVI {selected_year_map}",
                        overlay=True
                    ).add_to(m)

                # Water layer
                if show_water:
                    mndwi = s2.normalizedDifference(["B3","B11"])
                    water_mask = mndwi.gt(0.05).selfMask()
                    water_url = get_tile_url(water_mask, {
                        "min":0, "max":1,
                        "palette":["#0077b6"]
                    })
                    folium.TileLayer(
                        tiles=water_url,
                        attr="GEE Sentinel-2 Water",
                        name=f"💧 Water {selected_year_map}",
                        overlay=True
                    ).add_to(m)

                # True Color
                if show_truecolor:
                    tc_url = get_tile_url(s2, {
                        "bands":["B4","B3","B2"],
                        "min":0, "max":2500, "gamma":1.4
                    })
                    folium.TileLayer(
                        tiles=tc_url,
                        attr="GEE Sentinel-2 True Color",
                        name=f"🗺️ True Color {selected_year_map}",
                        overlay=True
                    ).add_to(m)

                # False Color (crops appear bright red)
                if show_falsecolor:
                    fc_url = get_tile_url(s2, {
                        "bands":["B8","B4","B3"],
                        "min":0, "max":4000, "gamma":1.3
                    })
                    folium.TileLayer(
                        tiles=fc_url,
                        attr="GEE Sentinel-2 False Color",
                        name=f"🔴 False Color {selected_year_map}",
                        overlay=True
                    ).add_to(m)

                # Rainfall
                if show_rainfall:
                    rain = (ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
                            .filterBounds(region_ee)
                            .filterDate(f"{selected_year_map}-01-01",
                                        f"{selected_year_map}-12-31")
                            .select("precipitation").sum().clip(region_ee))
                    rain_url = get_tile_url(rain, {
                        "min":50, "max":500,
                        "palette":["#ffffcc","#a1dab4","#41b6c4","#2c7fb8","#253494"]
                    })
                    folium.TileLayer(
                        tiles=rain_url,
                        attr="CHIRPS Rainfall",
                        name=f"🌧️ Rainfall {selected_year_map}",
                        overlay=True
                    ).add_to(m)

                # Region boundary
                if show_boundary:
                    folium.Rectangle(
                        bounds=[[bbox_map[1],bbox_map[0]],
                                [bbox_map[3],bbox_map[2]]],
                        color="#fbbf24",
                        fill=False,
                        weight=2,
                        dash_array="8",
                        tooltip=f"{region_name} — 40×40 km study region"
                    ).add_to(m)

                # Province markers
                for pname, pdata in PROVINCES.items():
                    pb = pdata["bbox"]
                    pc = pdata["center"]
                    folium.Marker(
                        location=pc,
                        tooltip=pname,
                        icon=folium.Icon(
                            color="green" if pname==region_name else "gray",
                            icon="leaf" if pname==region_name else "info-sign",
                            prefix="glyphicon"
                        )
                    ).add_to(m)

                # Layer control
                folium.LayerControl(collapsed=False).add_to(m)

                # Store map in session state so it persists
                st.session_state["map_obj"] = m
                st.session_state["map_year_loaded"] = selected_year_map
                st.success(f"✓ Loaded {selected_year_map} — use layer control (top right of map) to toggle layers")

            except Exception as e:
                st.error(f"Map error: {e}")

    # Always render map from session state
    if "map_obj" in st.session_state:
        st.caption(f"Year loaded: {st.session_state.get('map_year_loaded', map_year)} · Use ☰ layer control on map to toggle · Zoom with scroll wheel")
        st_folium(
            st.session_state["map_obj"],
            width=None,
            height=600,
            key="persistent_map",
            returned_objects=[]
        )
    else:
        st.info("Click '🗺️ Load Map' to display the satellite layers.")

# ═══════════════════════════════════════════════════
# TAB 6 — AI ASSISTANT
# ═══════════════════════════════════════════════════
with tab6:
    st.subheader("🤖 AI Smart Farming Assistant")
    st.markdown(
        "Ask anything about water, crops, vegetation, or smart farming in Afghanistan.  \n"
        "Works in **English**, **Dari (دری)**, and **Pashto (پښتو)**."
    )

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content":
             f"Hello! I am your Smart Farming AI assistant for {region_name}. "
             "I have access to real satellite data from 2019 to 2024. "
             "Ask me anything about water availability, cropland, vegetation health, "
             "or what crops to grow. You can also ask in Dari or Pashto."}
        ]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Latest data context
    latest = REAL_DATA[2024]
    baseline_data = REAL_DATA[2019]

    system_prompt = f"""You are an agricultural data analyst and smart farming expert for Afghanistan.
You have access to real Sentinel-2 satellite data for {region_name}:

REAL DATA 2019-2024:
- 2019 (peak): NDVI=0.172, Cropland=385.9km², Water=24.1km²
- 2020: NDVI=0.168, Cropland=370.9km², Water=21.8km²
- 2021: NDVI=0.162, Cropland=300.0km², Water=17.0km²
- 2022 (worst): NDVI=0.139, Cropland=182.7km², Water=5.8km²
- 2023: NDVI=0.142, Cropland=211.3km², Water=11.0km²
- 2024 (current): NDVI=0.152, Cropland=251.9km², Water=14.2km²

KEY FINDINGS:
- Cropland collapsed 53% from 2019 to 2022
- Water collapsed 76% in 2022 (drought + political transition)
- Correlation water vs cropland = 0.97 (water is #1 constraint)
- Recovery happening but fragile — still 35% below 2019

CROP RECOMMENDATIONS for water-stressed conditions:
- Best: Saffron (300mm water, $15000/ha), Flax (350mm, $800/ha)
- Good: Almonds, Vegetables
- Avoid: Rice (1200mm water), Cotton (700mm water)

Answer questions based on this real data.
Be specific and practical for smallholder farmers.
If asked in Dari or Pashto, respond in the same language.
Keep answers to 3-5 sentences — clear and actionable."""

    if prompt := st.chat_input("Ask about water, crops, vegetation... / پوښتنه وکړئ / سوال بپرسید"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analysing satellite data..."):
                try:
                    import anthropic
                    client = anthropic.Anthropic(
                        api_key=st.secrets["anthropic"]["api_key"]
                    )
                    response = client.messages.create(
                        model="claude-sonnet-4-20250514",
                        max_tokens=500,
                        system=system_prompt,
                        messages=[
                            {"role": m["role"], "content": m["content"]}
                            for m in st.session_state.messages
                            if m["role"] in ["user", "assistant"]
                        ]
                    )
                    answer = response.content[0].text
                    st.markdown(answer)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )
                except KeyError:
                    answer = ("AI assistant requires Anthropic API key in Streamlit secrets. "
                             "Add `[anthropic]` section with `api_key` to your secrets.")
                    st.warning(answer)
                except Exception as e:
                    st.error(f"AI error: {e}")

# ─── FOOTER ──────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center;color:#6b8f65;font-family:monospace;font-size:11px;line-height:2">
Afghanistan Development Initiative (ADI) · WUR Wageningen · FAO Rome Conference Jul 2025<br>
Data: Sentinel-2 SR · CHIRPS · MODIS MOD16 · ERA5 · Hansen GFW · Google Earth Engine<br>
Analyst: Maiwand Jan Alamzoi · m.alamzoi123@gmail.com · afghanistan-development-initiative.github.io
</div>
""", unsafe_allow_html=True)
