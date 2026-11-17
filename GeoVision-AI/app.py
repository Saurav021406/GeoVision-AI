"""
GeoVision AI - Streamlit website
Run:  streamlit run app.py
"""
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import geovision_pipeline as gv

st.set_page_config(page_title="GeoVision AI", page_icon="🛰️", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1280px;}
.gv-title {font-size: 2.2rem; font-weight: 700; letter-spacing: -0.01em; margin: 0; line-height: 1.15;}
.gv-sub {color: #8b97ad; margin: .15rem 0 1rem 0;}
.chip {display: inline-flex; align-items: center; gap: .4rem; padding: .16rem .65rem; border-radius: 999px;
       background: #101828; border: 1px solid #1e2a3d; font-size: .82rem; margin: .15rem .35rem .15rem 0;}
.chip i {width: .7rem; height: .7rem; border-radius: 3px; display: inline-block;}
.note {border-left: 3px solid #f2a93b; background: rgba(242,169,59,.08); padding: .55rem .9rem;
       border-radius: 4px; font-size: .86rem; margin-bottom: .8rem;}
div[data-testid="stMetric"] {background: #101828; border: 1px solid #1e2a3d; border-radius: 8px; padding: .7rem .9rem;}
</style>
""", unsafe_allow_html=True)

EXT = ["jpg", "jpeg", "png", "tif", "tiff"]
TEAL = "#29d9c2"
HEX = {name: "#%02x%02x%02x" % gv.DISASTER_COLORS[c] for c, name in gv.DISASTER_NAMES.items() if c in gv.DISASTER_COLORS}

MODES = {
    "🛰️ Full scan": {"tasks": ["classifier", "disaster", "flood", "change"], "after": "optional",
                      "help": "Runs every available model. Add an 'after' image to also detect change."},
    "🌪️ Disaster detection": {"tasks": ["disaster", "change"], "after": "optional",
                               "help": "Finds flood, wildfire, landslide, earthquake damage, drought and storm damage."},
    "🌍 Land cover": {"tasks": ["classifier"], "after": None,
                      "help": "Classifies the scene into 10 land-cover types and shows where the model looked."},
    "🌊 Flood mapping": {"tasks": ["flood"], "after": None,
                         "help": "Dedicated binary flood segmentation with an adjustable threshold."},
    "🔄 Change detection": {"tasks": ["change"], "after": "required",
                            "help": "Upload a before and an after image to see what changed."},
}
SCENES = {"Flood": 1, "Wildfire": 2, "Landslide": 3, "Earthquake damage": 4, "Drought": 5,
          "Cyclone / storm damage": 6, "Clean (no disaster)": 0}


# --------------------------------------------------------------------- helpers
def stretch(fn, *args, **kwargs):
    """Call a Streamlit element so it fills its column, whatever the Streamlit version."""
    for extra in ({"width": "stretch"}, {"use_container_width": True}, {"use_column_width": True}, {}):
        try:
            return fn(*args, **kwargs, **extra)
        except Exception:  # noqa: BLE001
            continue


def show(img, caption=None):
    if isinstance(img, np.ndarray) and img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    stretch(st.image, img, caption=caption)


def u8(arr):
    return (np.clip(arr, 0, 1) * 255).astype(np.uint8)


def fit(mask, shape):
    if mask.shape[:2] == shape[:2]:
        return mask
    return cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)


def legend(names):
    chips = "".join(f'<span class="chip"><i style="background:{HEX.get(n, TEAL)}"></i>{n}</span>' for n in names)
    st.markdown(chips, unsafe_allow_html=True)


def bar_chart(data, colors=None):
    df = pd.DataFrame({"Class": list(data.keys()), "Area %": [round(float(v), 2) for v in data.values()]})
    colors = colors or {}
    try:
        import altair as alt
        chart = (alt.Chart(df).mark_bar()
                 .encode(x=alt.X("Area %:Q"), y=alt.Y("Class:N", sort="-x", title=None),
                         color=alt.Color("Class:N", legend=None,
                                         scale=alt.Scale(domain=list(df["Class"]),
                                                         range=[colors.get(c, TEAL) for c in df["Class"]])),
                         tooltip=["Class", "Area %"])
                 .properties(height=30 * len(df) + 30))
        stretch(st.altair_chart, chart)
    except Exception:  # noqa: BLE001
        st.bar_chart(df.set_index("Class"))


def table(df):
    try:
        stretch(st.dataframe, df, hide_index=True)
    except Exception:  # noqa: BLE001
        st.dataframe(df)


@st.cache_resource(show_spinner="Loading models…")
def get_pipeline():
    try:
        if "GEOVISION_HF_REPO" in st.secrets:
            os.environ.setdefault("GEOVISION_HF_REPO", st.secrets["GEOVISION_HF_REPO"])
    except Exception:  # noqa: BLE001
        pass
    return gv.GeoVisionPipeline()


@st.cache_resource
def eurosat_dataset(root):
    try:
        from torchvision.datasets import EuroSAT
        folder = Path(root) / "data" / "eurosat"
        return EuroSAT(root=str(folder), download=False) if folder.exists() else None
    except Exception:  # noqa: BLE001
        return None


pipe = get_pipeline()


# --------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### 🛰️ GeoVision AI")
    mode = st.radio("Analysis mode", list(MODES.keys()), index=0)
    st.caption(MODES[mode]["help"])
    with st.expander("⚙️ Settings"):
        tiled = st.checkbox("Tile large images", value=False,
                            help="Cuts the image into 256×256 tiles instead of shrinking it. "
                                 "Better for big satellite scenes, slower on CPU.")
        flood_thr = st.slider("Flood threshold", 0.10, 0.90, 0.50, 0.05,
                              help="Lower = more area marked as flood.")
        change_thr = st.slider("Change threshold", 0.10, 0.90, 0.50, 0.05)
        opacity = st.slider("Overlay opacity", 0.10, 1.00, 0.55, 0.05)
    st.markdown("**Models**")
    for task, label in gv.TASK_LABELS.items():
        st.markdown(f"{'✅' if pipe.available(task) else '⚠️'} {label}")
        if task in pipe.errors:
            st.caption(pipe.errors[task])
    st.caption(f"Running on: {str(pipe.device).upper()}")

st.markdown('<p class="gv-title">🛰️ GeoVision AI</p>', unsafe_allow_html=True)
st.markdown('<p class="gv-sub">Satellite image analysis: land cover, multi-disaster detection, '
            'flood mapping and change detection.</p>', unsafe_allow_html=True)

tab_run, tab_perf, tab_about = st.tabs(["🔍 Analyze", "📊 Model performance", "ℹ️ About"])


# ------------------------------------------------------------------- input UI
def get_inputs(mode_name):
    cfg = MODES[mode_name]
    after_mode = cfg["after"]
    key = f"m{list(MODES).index(mode_name)}"          # unique widget keys per mode
    source = st.radio("Image source", ["Upload my own", "Try a sample scene"], horizontal=True, key=f"src_{key}")
    inp = {"before": None, "after": None, "gt": {}, "label": ""}

    if source == "Upload my own":
        if after_mode:
            c1, c2 = st.columns(2)
            f1 = c1.file_uploader("Before / current image", type=EXT, key=f"up1_{key}")
            f2 = c2.file_uploader("After image" + (" (optional)" if after_mode == "optional" else ""),
                                  type=EXT, key=f"up2_{key}")
        else:
            f1 = st.file_uploader("Satellite image", type=EXT, key=f"up1_{key}")
            f2 = None
        try:
            if f1 is not None:
                inp["before"], inp["label"] = gv.load_pil(f1), f1.name
            if f2 is not None:
                inp["after"] = gv.load_pil(f2)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not read that image: {exc}")
        return inp

    st.session_state.setdefault("seed", 11)
    c1, c2 = st.columns([3, 1])
    scene = None
    if mode_name in ("🛰️ Full scan", "🌪️ Disaster detection"):
        scene = c1.selectbox("Scene type", list(SCENES.keys()))
    if c2.button("🎲 New random scene", key=f"dice_{key}"):
        st.session_state["seed"] += 1
    seed = st.session_state["seed"]

    if mode_name in ("🛰️ Full scan", "🌪️ Disaster detection"):
        img, mask = gv.make_disaster_scene(SCENES[scene], seed=seed)
        inp.update(before=img, gt={"disaster": mask}, label=f"sample: {scene} (seed {seed})")
    elif mode_name == "🌊 Flood mapping":
        img, mask = gv.make_flood_scene(seed=seed)
        inp.update(before=img, gt={"flood": mask}, label=f"sample: flood scene (seed {seed})")
    elif mode_name == "🔄 Change detection":
        a, b, mask = gv.make_change_pair(seed=seed)
        inp.update(before=a, after=b, gt={"change": mask}, label=f"sample: change pair (seed {seed})")
    else:  # land cover
        ds = eurosat_dataset(str(pipe.root))
        if ds is None:
            st.info("No EuroSAT dataset found on this machine. Upload a photo instead.")
        else:
            img, lab = ds[seed % len(ds)]
            inp.update(before=img.convert("RGB"), gt={"label": ds.classes[lab]},
                       label=f"sample: EuroSAT #{seed % len(ds)}")
    return inp


# -------------------------------------------------------------------- results
def render(R, op):
    res, gt = R["res"], R.get("gt") or {}
    arr = res["image"]
    items = []
    if "classification" in res:
        c = res["classification"]
        items.append(("Land cover", c["label"], f"{c['confidence']:.0%} confidence"))
    if "disaster" in res:
        d = res["disaster"]
        items += [("Dominant hazard", d["dominant"], None),
                  ("Affected area", f"{d['affected_pct']:.1f}%", f"{d['severity']} severity")]
    if "flood" in res:
        items.append(("Flooded area", f"{res['flood']['percentage']:.1f}%", f"{res['flood']['severity']} severity"))
    if "change" in res:
        items.append(("Changed area", f"{res['change']['percentage']:.1f}%", None))
    if not items:
        st.warning("None of the models needed for this mode are available.")
        return
    for col, (label, value, delta) in zip(st.columns(len(items)), items):
        col.metric(label, value, delta, delta_color="off")

    lines = []
    if "classification" in res:
        c = res["classification"]
        lines.append(f"**Land cover:** looks like **{c['label']}** ({c['confidence']:.0%} confidence).")
    if "disaster" in res:
        d = res["disaster"]
        if d["dominant"] == "No disaster detected":
            lines.append("**Disaster scan:** no hazard pattern covers more than 1% of the scene.")
        else:
            lines.append(f"**Disaster scan:** largest hazard is **{d['dominant']}** "
                         f"({d['percentages'][d['dominant']]:.1f}% of the scene); {d['affected_pct']:.1f}% "
                         f"of the area is affected in total, so severity is **{d['severity']}**.")
    if "flood" in res:
        lines.append(f"**Flood model:** {res['flood']['percentage']:.1f}% of the scene is flooded.")
    if "change" in res:
        lines.append(f"**Change detection:** {res['change']['percentage']:.1f}% of the area changed.")
    st.info("\n\n".join(lines))

    if "disaster" in res:
        d = res["disaster"]
        st.markdown("### 🌪️ Disaster detection")
        cols = st.columns(4 if "disaster" in gt else 3)
        with cols[0]:
            show(u8(arr), "Input")
        with cols[1]:
            show(gv.overlay(arr, gv.colorize(d["mask"]), op), "Detected hazards")
        with cols[2]:
            show(gv.colorize(d["mask"]), "Hazard map")
        if "disaster" in gt:
            with cols[3]:
                show(gv.colorize(fit(gt["disaster"], arr.shape)), "Ground truth (sample)")
        present = [n for n, v in d["percentages"].items() if n != "Background" and v >= 0.5]
        legend(present if present else ["No hazard above 0.5%"])
        hazards = {k: v for k, v in d["percentages"].items() if k != "Background"}
        bar_chart(hazards, HEX)

    if "flood" in res:
        f = res["flood"]
        st.markdown("### 🌊 Flood mapping")
        cols = st.columns(4 if "flood" in gt else 3)
        with cols[0]:
            show(u8(arr), "Input")
        with cols[1]:
            show(gv.overlay(arr, gv.colorize_binary(f["mask"], (30, 100, 255)), op), "Flooded areas")
        with cols[2]:
            show(gv.colorize_binary(f["mask"], (30, 100, 255)), "Flood mask")
        if "flood" in gt:
            with cols[3]:
                show(gv.colorize_binary(fit(gt["flood"], arr.shape), (30, 100, 255)), "Ground truth (sample)")

    if "classification" in res:
        c = res["classification"]
        st.markdown("### 🌍 Land-cover classification")
        c1, c2, c3 = st.columns([1, 1, 1.3])
        with c1:
            show(u8(arr), "Input")
        with c2:
            if "cam" in c:
                show(gv.heatmap_overlay(arr, c["cam"]), "Grad-CAM: where the model looked")
        with c3:
            bar_chart({n: p * 100 for n, p in c["top"]})
            if "label" in gt:
                st.caption(f"True label of this sample: **{gt['label']}**")

    if "change" in res:
        ch = res["change"]
        st.markdown("### 🔄 Change detection")
        cols = st.columns(5 if "change" in gt else 4)
        with cols[0]:
            show(u8(arr), "Before")
        with cols[1]:
            show(u8(res["image_after"]), "After")
        with cols[2]:
            show(gv.overlay(res["image_after"], gv.colorize_binary(ch["mask"]), op), "Detected change")
        with cols[3]:
            show(gv.colorize_binary(ch["mask"]), "Change mask")
        if "change" in gt:
            with cols[4]:
                show(gv.colorize_binary(fit(gt["change"], arr.shape)), "Ground truth (sample)")
    if "delta" in res:
        st.markdown("**How hazard areas differ between the two images**")
        before_p, after_p = res["disaster"]["percentages"], res["disaster_after"]["percentages"]
        df = pd.DataFrame([{"Hazard": k, "Before %": round(before_p[k], 2), "After %": round(after_p[k], 2),
                            "Difference (pp)": round(v, 2)} for k, v in res["delta"].items()])
        table(df)

    st.markdown("---")
    rep = gv.build_report(res, {"mode": R["mode"], "source": R["label"]})
    b1, b2, b3 = st.columns(3)
    b1.download_button("⬇️ Report (JSON)", json.dumps(rep, indent=2), "geovision_report.json",
                       "application/json", key="dl_json")
    if "disaster" in res:
        csv_df = pd.DataFrame({"class": list(res["disaster"]["percentages"].keys()),
                               "area_percent": [round(v, 3) for v in res["disaster"]["percentages"].values()]})
        b2.download_button("⬇️ Class areas (CSV)", csv_df.to_csv(index=False), "geovision_areas.csv",
                           "text/csv", key="dl_csv")
        main_png = gv.overlay(arr, gv.colorize(res["disaster"]["mask"]), op)
    elif "flood" in res:
        main_png = gv.overlay(arr, gv.colorize_binary(res["flood"]["mask"], (30, 100, 255)), op)
    elif "change" in res:
        main_png = gv.overlay(res["image_after"], gv.colorize_binary(res["change"]["mask"]), op)
    elif "classification" in res and "cam" in res["classification"]:
        main_png = gv.heatmap_overlay(arr, res["classification"]["cam"])
    else:
        main_png = u8(arr)
    b3.download_button("⬇️ Result image (PNG)", gv.to_png_bytes(main_png), "geovision_result.png",
                       "image/png", key="dl_png")


# ------------------------------------------------------------------ Analyze tab
with tab_run:
    st.markdown('<div class="note"><b>Research prototype.</b> The disaster, flood and change models were trained on '
                'synthetic scenes, so results on real satellite photos are indicative only. '
                'Use "Try a sample scene" to see the models on the kind of data they know.</div>',
                unsafe_allow_html=True)
    if pipe.n_loaded == 0:
        st.error("No model checkpoints were found. Put the .pt files under checkpoints/, segmentation/, "
                 "change_detection/ and multi_disaster/ next to this app, or set GEOVISION_ROOT / GEOVISION_HF_REPO.")

    inp = get_inputs(mode)
    needs_after = MODES[mode]["after"] == "required"
    ready = inp["before"] is not None and (inp["after"] is not None or not needs_after)

    if st.button("▶ Run analysis", type="primary", disabled=not ready):
        wanted = [t for t in MODES[mode]["tasks"] if pipe.available(t)]
        if not wanted:
            st.error("The models needed for this mode are not loaded (see the sidebar).")
        else:
            try:
                with st.spinner("Analyzing…"):
                    res = pipe.analyze(inp["before"], inp["after"], tasks=wanted, tiled=tiled,
                                       flood_threshold=flood_thr, change_threshold=change_thr)
                st.session_state["result"] = {"res": res, "mode": mode, "gt": inp["gt"], "label": inp["label"]}
            except Exception as exc:  # noqa: BLE001
                st.error(f"Analysis failed: {exc}")
    elif not ready:
        st.caption("Add an image (or pick a sample scene) to enable the Run button."
                   + (" This mode needs both a before and an after image." if needs_after else ""))

    R = st.session_state.get("result")
    if R and R["mode"] == mode:
        st.markdown("---")
        render(R, opacity)
    elif R:
        st.caption(f"The last result was for “{R['mode']}”. Switch back to see it, or run a new analysis.")


# --------------------------------------------------------------- Performance tab
with tab_perf:
    st.markdown("### Loaded checkpoints")
    rows = [{"Task": gv.TASK_LABELS[t], "File": i["file"], "Size (MB)": i["size_mb"],
             "Best epoch": i["epoch"],
             "Best validation score": None if i["val_metric"] is None else round(float(i["val_metric"]), 4)}
            for t, i in pipe.ckpt_info.items()]
    if rows:
        table(pd.DataFrame(rows))
    for t, err in pipe.errors.items():
        st.warning(f"{gv.TASK_LABELS[t]}: {err}")

    st.markdown("### Experiment log")
    log = pipe.root / "experiments" / "experiment_log.csv"
    if not log.exists():
        st.info("experiments/experiment_log.csv was not found, so no training results are shown.")
    else:
        df = pd.read_csv(log)
        kinds = {"exp": ("Classification", "Accuracy"), "seg": ("Flood segmentation", "IoU"),
                 "dis": ("Multi-disaster", "mIoU"), "cd": ("Change detection", "IoU")}
        df["prefix"] = df["experiment_id"].astype(str).str.extract(r"^([a-z]+)")[0]
        df = df[df["prefix"].isin(kinds)].copy()
        df["Task"] = df["prefix"].map(lambda p: kinds[p][0])
        df["Metric"] = df["prefix"].map(lambda p: kinds[p][1])
        df["Score (%)"] = (pd.to_numeric(df["test_acc"], errors="coerce") * 100).round(2)
        shown = df[["experiment_id", "Task", "model", "Metric", "Score (%)", "training_time_min"]].rename(
            columns={"experiment_id": "Experiment", "model": "Model", "training_time_min": "Train time (min)"})
        table(shown)
        chart_df = shown.dropna(subset=["Score (%)"])
        if not chart_df.empty:
            st.bar_chart(chart_df.set_index("Experiment")["Score (%)"])
        st.caption("Scores come straight from the experiment log. Segmentation and change-detection scores are on "
                   "synthetic test data.")


# ------------------------------------------------------------------- About tab
with tab_about:
    st.markdown("""
### What this does
GeoVision AI bundles four deep-learning models behind one website:

| Task | Model | Output |
|---|---|---|
| Land-cover classification | Hybrid ResNet50 + Swin-T | 1 of 10 EuroSAT classes + Grad-CAM |
| Multi-disaster detection | SegFormer-style transformer | per-pixel map of 6 hazard types |
| Flood mapping | SegFormer-style transformer | flood mask |
| Change detection | ChangeFormer-style transformer | change mask between two images |

### Hazard classes
""")
    legend([n for c, n in gv.DISASTER_NAMES.items() if c != 0])
    st.markdown("""
### Honest limitations
- Classification was trained on real EuroSAT images (64×64 Sentinel-2 tiles).
- Disaster, flood and change models were trained on **synthetic** scenes (coloured patches on a simple landscape).
  Expect them to work on similar scenes and to be unreliable on real photos until retrained on real data
  (for example xBD, FloodNet, LEVIR-CD).
- Severity labels are a simple rule of thumb (<1% none, <10% low, <30% moderate, otherwise severe),
  not an official damage assessment.
- Images larger than 1536 px are scaled down. Tiling processes the scene in 256×256 blocks.

### Run it yourself

    pip install -r requirements.txt
    streamlit run app.py
    python geovision_pipeline.py scene.jpg --out results/      # command line

Set `GEOVISION_ROOT` to the folder holding the checkpoints, or `GEOVISION_HF_REPO` to download them from Hugging Face.
""")
