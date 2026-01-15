import streamlit as st
import folium
from folium.raster_layers import ImageOverlay
from streamlit_folium import st_folium
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

from utils_preprocessed_64_sharp import (
    load_ground_truth,
    array_to_rgba_overlay,
    rescale_after_threshold,
    smooth_prediction,
    gamma_boost,
    BOUNDS,
)

st.set_page_config(layout="wide")

NOWCAST_ROOT = "/work/scratch-nopw2/mendrika/pancast-live"

st.markdown(
    """
    <style>
    .title { font-size: 36px; font-weight: 700; padding-bottom: 10px; }
    html, body { font-size: 22px; }
    div[data-testid="stWidgetLabel"] label {
        font-size: 26px !important;
        font-weight: 700 !important;
    }
    </style>

    <div class="title">PANCAST Interactive Nowcast Viewer</div>
    """,
    unsafe_allow_html=True,
)


col_date = st.columns(5)

with col_date[0]:
    year = st.selectbox("Year", range(2020, 2027), index=4)
with col_date[1]:
    month = st.selectbox("Month", range(1, 13), index=6)
with col_date[2]:
    day = st.selectbox("Day", range(1, 32), index=14)
with col_date[3]:
    hour = st.selectbox("Hour", range(24), index=12)
with col_date[4]:
    minute = st.selectbox("Minute", [0, 15, 30, 45], index=0)

t0 = datetime(year, month, day, hour, minute)

lead_time = st.radio(
    "Lead time (minutes)",
    options=[30, 60, 90, 120],
    horizontal=True,
)

prob_label = st.radio(
    "Probability of convection > (%)",
    options=["10", "30", "50", "70"],
    horizontal=True,
)

prob_thresh = int(prob_label) / 100.0
t_valid = t0 + timedelta(minutes=int(lead_time))

time_dict = {
    "year": f"{year:04d}",
    "month": f"{month:02d}",
    "day": f"{day:02d}",
    "hour": f"{hour:02d}",
    "minute": f"{minute:02d}",
}

origin_str = f"{time_dict['year']}{time_dict['month']}{time_dict['day']}_{time_dict['hour']}{time_dict['minute']}"

@st.cache_data(show_spinner=False)
def load_nowcast(origin_str, lead_time):
    fpath = (
        f"{NOWCAST_ROOT}/nowcasts_t{lead_time:03d}/"
        f"nowcast_t{lead_time:03d}_from_{origin_str}.npy"
    )
    if not Path(fpath).exists():
        raise FileNotFoundError(fpath)
    return np.load(fpath)

@st.cache_data(show_spinner=False)
def load_gt(time_dict, lead_time):
    return load_ground_truth(time_dict, lead_time)

data_available = True

try:
    pred = load_nowcast(origin_str, lead_time)
    gt = load_gt(time_dict, lead_time)
except Exception:
    data_available = False
    pred, gt = None, None


m = folium.Map(
    location=[5.0, 20.0],
    zoom_start=4.5,
    tiles="CartoDB positron",
)

info_html = f"""
<div style="
    position: absolute;
    top: 12px;
    left: 12px;
    z-index: 9999;
    background-color: rgba(255, 255, 255, 0.92);
    padding: 12px 16px;
    border-radius: 10px;
    border: 1px solid #cfcfcf;
    font-size: 18px;
    line-height: 1.5;
">
    <div><b>Nowcast origin (t₀):</b><br>{t0:%Y-%m-%d %H:%M} UTC</div>
    <div style="margin-top:6px;"><b>Valid time:</b><br>{t_valid:%Y-%m-%d %H:%M} UTC</div>
</div>
"""

m.get_root().html.add_child(folium.Element(info_html))


if data_available:
    # pred = rescale_after_threshold(pred, floor=0.1)
    # pred = gamma_boost(pred, gamma=0.9)
    # pred = smooth_prediction(pred, sigma=0.1)

    pred_img = array_to_rgba_overlay(
        pred,
        mask=pred > prob_thresh,
        cmap_name="viridis",
        vmin=0,
        vmax=1,
        alpha=1.0,
    )

    gt_img = array_to_rgba_overlay(
        gt.astype(float),
        mask=gt,
        cmap_name="Greys",
        vmin=0,
        vmax=1,
        alpha=1.0,
    )

    pred_layer = folium.FeatureGroup(
        name=f"Prediction t+{lead_time} min",
        show=True,
    )

    gt_layer = folium.FeatureGroup(
        name=f"Ground truth t+{lead_time} min",
        show=False,
    )

    ImageOverlay(pred_img, bounds=BOUNDS, opacity=1.0).add_to(pred_layer)
    ImageOverlay(gt_img, bounds=BOUNDS, opacity=1.0).add_to(gt_layer)

    pred_layer.add_to(m)
    gt_layer.add_to(m)

    folium.LayerControl(collapsed=True).add_to(m)
else:
    st.warning("No nowcast available for this time.")

# ------------------------
# Render
# ------------------------
st_folium(
    m,
    key="pancast_map",
    use_container_width=True,
    height=1024,
    returned_objects=[]
)
