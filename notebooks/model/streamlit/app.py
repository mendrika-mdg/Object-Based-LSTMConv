import streamlit as st
import folium
from folium.raster_layers import ImageOverlay
from streamlit_folium import st_folium

from utils import (
    load_models,
    load_input_tensor,
    load_ground_truth,
    scale_input,
    ensemble_predict,
    array_to_rgba_overlay,
    BOUNDS,
)

from datetime import datetime, timedelta

st.set_page_config(layout="wide")

st.markdown(
    """
    <style>
    .title {
        font-size: 32px;
        font-weight: 700;
        padding-bottom: 10px;
    }

    html, body {
        font-size: 20px;
    }

    label {
        font-size: 20px !important;
        font-weight: 600 !important;
    }

    div[data-baseweb="select"] * {
        font-size: 20px !important;
        font-weight: 600 !important;
    }

    div[role="radiogroup"] label {
        font-size: 20px !important;
        font-weight: 600 !important;
    }

    div[role="radiogroup"] span {
        font-size: 20px !important;
        font-weight: 600 !important;
    }

    input, textarea, button {
        font-size: 20px !important;
        font-weight: 600 !important;
    }

    .stRadio > label {
        font-size: 21px !important;
        font-weight: 700 !important;
    }
    </style>

    <div class="title">PANCAST Interactive Nowcast Viewer</div>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <div style="
        padding: 12px 16px;
        border-radius: 8px;
        background-color: #e8f2ff;
        border: 1px solid #c6dcff;
        font-size: 20px;
        line-height: 1.6;
    ">
        This portal visualises probabilistic nowcasts of convective storms for the next 2 hours.
        Select the <b>nowcast origin time (t₀)</b> and a <b>lead time</b> to explore predicted storm probabilities.
    </div>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <style>
    .controls {
        max-width: 100px;
        margin: 0 auto;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="controls">', unsafe_allow_html=True)

col_date = st.columns([1, 1, 1, 1, 1])

with col_date[0]:
    year = st.selectbox("Year", range(2004, 2025), index=20)
with col_date[1]:
    month = st.selectbox("Month", range(1, 13), index=6)
with col_date[2]:
    day = st.selectbox("Day", range(1, 32), index=14)
with col_date[3]:
    hour = st.selectbox("Hour", range(24), index=12)
with col_date[4]:
    minute = st.selectbox("Minute", [0, 15, 30, 45], index=0)

st.markdown('</div>', unsafe_allow_html=True)

t0 = datetime(
    int(year),
    int(month),
    int(day),
    int(hour),
    int(minute),
)

lead_time = st.radio(
    "Lead time (minutes)",
    options=[30, 60, 90, 120],
    horizontal=True,
)

prob_label = st.radio(
    "Probability of convection > (%)",
    options=[r" 5", "10", "20", "30"],
    horizontal=True,
    index=0,
)

prob_thresh = int(prob_label) / 100.0

t_valid = t0 + timedelta(minutes=int(lead_time))

time_key = (
    int(year),
    int(month),
    int(day),
    int(hour),
    int(minute),
)

@st.cache_resource
def cached_models(lt):
    return load_models(int(lt))

@st.cache_data
def run_all_lead_times(time_key):
    y, m, d, h, mi = time_key
    td = {
        "year": f"{y:04d}",
        "month": f"{m:02d}",
        "day": f"{d:02d}",
        "hour": f"{h:02d}",
        "minute": f"{mi:02d}",
    }

    X = load_input_tensor(td)
    X = scale_input(X).unsqueeze(0)

    preds = {}
    gts = {}

    for lt in [30, 60, 90, 120]:
        models = cached_models(lt)
        pred = ensemble_predict(models, X)
        gt = load_ground_truth(td, lt)

        preds[lt] = pred.cpu().numpy()
        gts[lt] = gt

    return preds, gts

@st.cache_data
def build_base_map():
    m = folium.Map(
        location=AFRICA_VIEW_CENTER,
        zoom_start=AFRICA_VIEW_ZOOM,
        tiles=None,
        max_bounds=True,
    )

    m.options["maxBounds"] = AFRICA_VIEW_BOUNDS
    m.options["minZoom"] = 3
    m.options["maxZoom"] = 8
    folium.TileLayer("CartoDB positron", name="Light", show=True).add_to(m)
    return m

AFRICA_VIEW_CENTER = [5.0, 20.0]
AFRICA_VIEW_ZOOM = 4.47

AFRICA_VIEW_BOUNDS = [
    [-45.0, -30.0],
    [45.0, 80.0],
]

preds = None
gts = None

try:
    preds, gts = run_all_lead_times(time_key)
except:
    pass

m = build_base_map()


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
    font-size: 16px;
    line-height: 1.5;
    box-shadow: 0 2px 6px rgba(0,0,0,0.15);
">
    <div><b>Nowcast origin (t₀):</b><br>{t0:%Y-%m-%d %H:%M} UTC</div>
    <div style="margin-top:6px;"><b>Valid time:</b><br>{t_valid:%Y-%m-%d %H:%M} UTC</div>
</div>
"""

m.get_root().html.add_child(folium.Element(info_html))


if preds is not None and int(lead_time) in preds:
    pred = preds[int(lead_time)]
    gt = gts[int(lead_time)]

    pred_img = array_to_rgba_overlay(
        pred,
        mask=pred > prob_thresh,
        cmap_name="Reds",
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

    pred_layer = folium.FeatureGroup(name=f"Prediction t+{lead_time} min", show=True)
    gt_layer = folium.FeatureGroup(name=f"Ground truth t+{lead_time} min", show=True)

    ImageOverlay(pred_img, bounds=BOUNDS, opacity=1.0).add_to(pred_layer)
    ImageOverlay(gt_img, bounds=BOUNDS, opacity=1.0).add_to(gt_layer)

    pred_layer.add_to(m)
    gt_layer.add_to(m)

folium.LayerControl(collapsed=True).add_to(m)




st_folium(m, use_container_width=True, height=1024)
