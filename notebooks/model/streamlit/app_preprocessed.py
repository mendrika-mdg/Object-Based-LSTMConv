import streamlit as st
import folium
from folium.raster_layers import ImageOverlay
from streamlit_folium import st_folium
from datetime import datetime, timedelta

from utils_preprocessed_64_sharp import (
    load_models,
    load_input_tensor,
    load_ground_truth,
    scale_input,
    ensemble_predict,
    array_to_rgba_overlay,
    rescale_after_threshold,
    smooth_prediction,
    gamma_boost,
    BOUNDS,
)

st.set_page_config(layout="wide")

# Persistent state
if "cached_time_key" not in st.session_state:
    st.session_state.cached_time_key = None
    st.session_state.cached_preds = None
    st.session_state.cached_gts = None

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

time_key = (year, month, day, hour, minute)

@st.cache_resource
def cached_models(lt):
    return load_models(lt)

@st.cache_data(show_spinner=False)
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

    preds, gts = {}, {}

    for lt in [30, 60, 90, 120]:
        models = cached_models(lt)
        preds[lt] = ensemble_predict(models, X).cpu().numpy()
        gts[lt] = load_ground_truth(td, lt)

    return preds, gts


# Attempt computation safely
data_available = True

if st.session_state.cached_time_key != time_key:
    try:
        preds, gts = run_all_lead_times(time_key)
        st.session_state.cached_preds = preds
        st.session_state.cached_gts = gts
        st.session_state.cached_time_key = time_key
    except:
        data_available = False
        preds, gts = None, None
else:
    preds = st.session_state.cached_preds
    gts = st.session_state.cached_gts
    if preds is None or gts is None:
        data_available = False


# Map is always shown
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


# Only add overlays if data exist
if data_available:
    pred = preds[int(lead_time)]
    gt = gts[int(lead_time)]

    pred = rescale_after_threshold(pred, floor=0.12)
    pred = gamma_boost(pred, gamma=0.6)
    pred = smooth_prediction(pred, sigma=0.8)

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


st_folium(
    m,
    key="pancast_map",
    use_container_width=True,
    height=1024,
    returned_objects=[]
)