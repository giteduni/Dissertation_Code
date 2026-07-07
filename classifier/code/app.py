from __future__ import annotations

import io

import numpy as np
import pandas as pd
import streamlit as st

import config
import predict

st.set_page_config(page_title="Finance Content Classifier", page_icon="▶️", layout="wide")

_STATIC_CSS = """
html, body, [class*="css"], .stApp, [data-testid="stAppViewContainer"],
[data-testid="stSidebar"], button, input, textarea, select{
  font-family:'Roboto', system-ui, -apple-system, 'Segoe UI', sans-serif !important;
}
.stApp, [data-testid="stAppViewContainer"]{ background:var(--yt-bg); color:var(--yt-text); }
/* Clear Streamlit's fixed top toolbar so the theme toggle row is never clipped. */
.block-container{ padding-top:5rem; }
[data-testid="stHeader"]{ background:var(--yt-header-bg); backdrop-filter:blur(6px); }
[data-testid="stToggle"], [data-testid="stCheckbox"]{ margin-top:.25rem; }
[data-testid="stSidebar"]{ background:var(--yt-surface); border-right:1px solid var(--yt-border); }
[data-testid="stSidebar"] *{ color:var(--yt-text); }

h1,h2,h3,h4{ font-weight:500; letter-spacing:-.01em; color:var(--yt-text); }
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li{ color:var(--yt-text); }
.stCaption, [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p{
  color:var(--yt-text-2) !important; }

/* Branded header */
.yt-header{ display:flex; align-items:center; gap:14px; margin:.1rem 0 .1rem; }
.yt-mark{ background:var(--yt-red); color:#fff; min-width:46px; height:32px; border-radius:9px;
  display:flex; align-items:center; justify-content:center; font-size:17px;
  box-shadow:0 2px 6px rgba(0,0,0,.45); }
.yt-title{ font-size:1.55rem; font-weight:700; color:var(--yt-text); line-height:1.12; }
.yt-sub{ color:var(--yt-text-2); font-size:.83rem; margin:.25rem 0 1.2rem; line-height:1.5; }

/* Theme toggle pinned to the top-right */
[data-testid="stToggle"], [data-testid="stCheckbox"]{ justify-content:flex-end; }

/* Primary actions -> YouTube red pill */
.stButton>button, [data-testid="stFormSubmitButton"] button, [data-testid="stDownloadButton"] button{
  background:var(--yt-red); color:#fff; border:none; border-radius:18px; font-weight:500;
  padding:.5rem 1.25rem; transition:background .15s ease, transform .05s ease;
}
.stButton>button:hover, [data-testid="stFormSubmitButton"] button:hover,
[data-testid="stDownloadButton"] button:hover{ background:var(--yt-red-h); color:#fff; }
.stButton>button:active, [data-testid="stFormSubmitButton"] button:active{ transform:translateY(1px); }
.stButton>button:focus{ box-shadow:none !important; }

/* Inputs */
[data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea,
[data-testid="stNumberInput"] input{
  background:var(--yt-surface-2); color:var(--yt-text);
  border:1px solid var(--yt-border); border-radius:10px;
}
[data-testid="stTextInput"] input:focus, [data-testid="stTextArea"] textarea:focus,
[data-testid="stNumberInput"] input:focus{
  border-color:var(--yt-blue); box-shadow:0 0 0 1px var(--yt-blue);
}
[data-testid="stWidgetLabel"] label, [data-testid="stWidgetLabel"] p{
  color:var(--yt-text-2) !important; font-weight:500; }

/* Form + uploader as cards */
[data-testid="stForm"]{ background:var(--yt-surface); border:1px solid var(--yt-border);
  border-radius:14px; padding:1.15rem 1.25rem; }
[data-testid="stFileUploader"]{ background:var(--yt-surface); border:1px solid var(--yt-border);
  border-radius:14px; padding:.5rem .6rem; }
[data-testid="stFileUploaderDropzone"]{ background:var(--yt-surface-2);
  border:1px dashed var(--yt-border); border-radius:11px; }

/* Metric cards */
[data-testid="stMetric"]{ background:var(--yt-surface); border:1px solid var(--yt-border);
  border-radius:13px; padding:14px 18px; }
[data-testid="stMetricValue"]{ color:var(--yt-text); font-weight:700; }
[data-testid="stMetricLabel"] p{ color:var(--yt-text-2) !important; }

/* DataFrame + tables */
[data-testid="stDataFrame"], [data-testid="stTable"]{ border:1px solid var(--yt-border);
  border-radius:13px; overflow:hidden; }

/* Sidebar radio -> tidy nav */
[data-testid="stSidebar"] [role="radiogroup"]{ gap:2px; }

/* Alerts a touch rounder */
[data-testid="stAlert"]{ border-radius:11px; }

/* Slider value/track use the red primaryColor from theme automatically */
"""

_FONT_IMPORT = (
    "@import url('https://fonts.googleapis.com/css2?"
    "family=Roboto:wght@400;500;700&display=swap');"
)


def _theme_css(dark: bool) -> str:
    """Return the full <style> block for the chosen theme (dark or light)."""
    if dark:
        palette = (
            "--yt-bg:#0f0f0f; --yt-surface:#212121; --yt-surface-2:#181818; --yt-hover:#303030;"
            "--yt-border:rgba(255,255,255,.12); --yt-text:#f1f1f1; --yt-text-2:#aaaaaa;"
            "--yt-track:#303030; --yt-blue:#3ea6ff; --yt-header-bg:rgba(15,15,15,.85);"
        )
    else:
        palette = (
            "--yt-bg:#ffffff; --yt-surface:#f9f9f9; --yt-surface-2:#ffffff; --yt-hover:#e5e5e5;"
            "--yt-border:rgba(0,0,0,.12); --yt-text:#0f0f0f; --yt-text-2:#606060;"
            "--yt-track:#e5e5e5; --yt-blue:#065fd4; --yt-header-bg:rgba(255,255,255,.9);"
        )
    root = ":root{ --yt-red:#ff0000; --yt-red-h:#cc0000; --yt-green:#2ba640; " + palette + " }"
    return "<style>" + _FONT_IMPORT + root + _STATIC_CSS + "</style>"


@st.cache_resource(show_spinner="Loading models…")
def _load():
    """Force the inference bundle to load once per server process."""
    predict.classify("warmup", "warmup", explain=False)  
    return predict._bundle()


def _models_ready() -> bool:
    try:
        _load()
        return True
    except FileNotFoundError:
        return False


def _gauge(p: float) -> None:
    """A YouTube-style probability bar for P(misleading) (red fill = misleading)."""
    pct = int(round(p * 100))
    dark = st.session_state.get("dark_mode", True)
    track = "#303030" if dark else "#e5e5e5"
    label_color = "#ffffff" if dark else "#0f0f0f"
    color = "#ff0000" if p >= config.STAGE2_MISLEADING_THRESHOLD else "#2ba640"
    st.markdown(
        f"""
        <div style="margin:.35rem 0 .25rem;">
          <div style="background:{track};border-radius:999px;height:30px;width:100%;
                      position:relative;overflow:hidden;">
            <div style="background:{color};width:{pct}%;height:100%;border-radius:999px;
                        transition:width .35s ease;"></div>
            <div style="position:absolute;inset:0;display:flex;align-items:center;
                        justify-content:center;font-weight:700;color:{label_color};
                        font-family:Roboto,sans-serif;font-size:.92rem;
                        text-shadow:0 1px 2px rgba(0,0,0,.18);">P(misleading) = {p:.3f}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _verdict_badge(label: str) -> str:
    """Return an HTML pill chip for the Stage-2 verdict."""
    if label == "Misleading":
        bg, icon, text = "#ff0000", "✘", "Misleading"
    else:
        bg, icon, text = "#2ba640", "✔", "Educational"
    return (
        f'<span style="display:inline-flex;align-items:center;gap:7px;background:{bg};'
        f'color:#fff;font-weight:700;font-family:Roboto,sans-serif;padding:6px 16px;'
        f'border-radius:999px;font-size:.92rem;letter-spacing:.02em;">{icon} {text}</span>'
    )


def _pretty_feature(name: str) -> str:
    if name.startswith("w:"):
        return f'word "{name[2:]}"'
    if name.startswith("c:"):
        return f'char "{name[2:].strip()}"'
    if name.startswith("lex_"):
        return f"codebook marker · {name[4:]}"
    return f"signal · {name}"


if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = True
_, _tb_right = st.columns([0.74, 0.26])
with _tb_right:
    dark_mode = st.toggle(
        "🌙 Dark mode" if st.session_state["dark_mode"] else "☀️ Light mode",
        key="dark_mode",
        help="Switch between light and dark theme.",
    )
st.markdown(_theme_css(dark_mode), unsafe_allow_html=True)

st.markdown(
    """
    <div class="yt-header">
      <div class="yt-mark">▶</div>
      <div class="yt-title">Misleading vs Educational YouTube Financial Conent Classifier</div>
    </div>
    <div class="yt-sub">
      Two-stage, text-only pipeline · Stage&nbsp;1 gates non-financial videos ·
      Stage&nbsp;2 scores a <b>calibrated</b> P(misleading) · all reported performance is
      out-of-sample (grouped CV + temporal hold-out) · MSc dissertation, University of Edinburgh.
    </div>
    """,
    unsafe_allow_html=True,
)

if not _models_ready():
    st.error("No trained models found in `models/`. Run `python train.py` first, then reload.")
    st.stop()

mode = st.sidebar.radio("Mode", ["Single video", "CSV batch"], index=0)
st.sidebar.markdown("### Thresholds")
_tuned_t1 = float(_load().stage1_threshold)
t1 = st.sidebar.slider("Stage 1 · P(non-financial) cutoff", 0.0, 1.0, _tuned_t1, 0.01,
                       help="Default is the macro-F1-tuned Stage-1 threshold from training.")
t2 = st.sidebar.slider("Stage 2 · P(misleading) cutoff", 0.0, 1.0,
                       float(config.STAGE2_MISLEADING_THRESHOLD), 0.01)
st.sidebar.caption(f"Headline Stage-2 model: **{config.PRIMARY_STAGE2_MODEL}**")



# Single video:
if mode == "Single video":
    st.subheader("Single video")
    with st.form("single"):
        title = st.text_input("Video title", "Turn $500 into $50,000 in 30 days - GUARANTEED crypto strategy!")
        description = st.text_area(
            "Video description",
            "Join my FREE training and inner circle. Only a few spots left - link below. "
            "Passive income while you sleep. Quit your 9-5 this month.",
            height=160,
        )
        c1, c2 = st.columns(2)
        view_count = c1.number_input("view_count (optional)", min_value=0, value=10000, step=100)
        like_count = c2.number_input("like_count (optional)", min_value=0, value=850, step=10)
        submitted = st.form_submit_button("Classify", type="primary")

    if submitted:
        res = predict.classify(title, description, view_count, like_count,
                               stage1_threshold=t1, stage2_threshold=t2, explain=True)

        # Stage 1
        s1 = res["stage1"]
        if s1["eligibility"] == "NON_FINANCIAL":
            st.warning(f"**Stage 1: NON-FINANCIAL** — P(non-financial) = {s1['p_non_financial']:.3f}. "
                       "Stage 2 skipped (only financial videos are eligible for the misleading/educational judgement).")
        else:
            st.success(f"**Stage 1: FINANCIAL** — P(financial) = {s1['p_financial']:.3f}")

            label = res["label"]
            st.markdown(
                '<div style="display:flex;align-items:center;gap:12px;margin:.5rem 0 .7rem;">'
                '<span style="font-size:1.1rem;font-weight:500;color:#f1f1f1;">Stage&nbsp;2 verdict</span>'
                f'{_verdict_badge(label)}</div>',
                unsafe_allow_html=True,
            )
            _gauge(res["p_misleading"])
            cc1, cc2 = st.columns(2)
            cc1.metric("Model A · XGBoost (interpretable)", f"{res['p_misleading_model_a']:.3f}")
            cc2.metric("Model B · Embeddings + LR", f"{res['p_misleading_model_b']:.3f}")

            st.markdown("#### Why — top contributing signals")
            tf = res["top_features"]
            if tf:
                tdf = pd.DataFrame([{
                    "signal": _pretty_feature(f["feature"]),
                    "pushes toward": f["direction"],
                    "contribution": round(f["contribution"], 4),
                    "value": round(f["value"], 4),
                } for f in tf])
                st.dataframe(tdf, hide_index=True, use_container_width=True)
                st.caption("Contributions are per-instance SHAP values from Model A "
                           "(positive → Misleading, negative → Educational).")
            else:
                st.info("No strong individual signals surfaced for this input.")



# Mode 1 - CSV batch

else:
    st.subheader("CSV batch scoring")
    st.write("Upload a CSV with at least **title** and **description** columns "
             "(header variants like `Title`/`Description` are handled). All original "
             "columns and row order are preserved; three columns are appended.")
    up = st.file_uploader("CSV file", type=["csv"])

    if up is not None:
        try:
            raw = pd.read_csv(up, dtype=str, keep_default_na=False, encoding="utf-8")
        except Exception:
            up.seek(0)
            raw = pd.read_csv(up, dtype=str, keep_default_na=False, encoding="latin-1")

        st.write(f"Loaded **{len(raw)}** rows × {raw.shape[1]} columns.")
        ov = predict.training_overlap(raw)
        if ov["is_in_sample"]:
            st.error(
                f"⚠ This file overlaps the training set "
                f"({ov['overlap_fraction']*100:.0f}% of rows match training videos). "
                "Metrics computed on it are **in-sample** and are not valid performance "
                "estimates. Use a held-out file (e.g. `data/demo_holdout.csv`) for an honest demo."
            )
        elif ov["overlap_fraction"] > 0:
            st.warning(f"{ov['overlap_fraction']*100:.0f}% of rows match training videos; "
                       "those individual predictions are in-sample.")

        with st.spinner("Scoring…"):
            scored = predict.classify_batch(raw, stage1_threshold=t1, stage2_threshold=t2)

        # Summary
        c1, c2, c3, c4 = st.columns(4)
        n_fin = int((scored["predicted_eligibility"] == "FINANCIAL").sum())
        c1.metric("Rows", len(scored))
        c2.metric("Financial", n_fin)
        c3.metric("Misleading", int((scored["predicted_label"] == "Misleading").sum()))
        c4.metric("Educational", int((scored["predicted_label"] == "Educational").sum()))

        st.markdown("#### Predicted label distribution")
        dist = scored["predicted_label"].replace("", "NON_FINANCIAL").value_counts()
        st.bar_chart(dist)

        
        label_col = predict.find_label_column(raw)
        if label_col is not None:
            ev = predict.evaluate_against_gold(scored, label_col)
            st.markdown("#### Evaluation on user-supplied labels")
            if ev is None:
                st.info(f"Found label column `{label_col}` but no rows with "
                        "Educational/Misleading gold labels to score against.")
            else:
                note = ("⚠ in-sample — not a valid performance estimate"
                        if ov["is_in_sample"] else "out-of-sample for these rows")
                st.write(f"Macro-F1 vs `{label_col}` on {ev['n_evaluated']} financial rows: "
                         f"**{ev['macro_f1']:.3f}**  ({note})")
                cm = pd.DataFrame(ev["confusion_matrix"],
                                  index=[f"true {l}" for l in ev["labels"]],
                                  columns=[f"pred {l}" for l in ev["labels"]])
                st.table(cm)

        st.markdown("#### Preview")
        st.dataframe(scored.head(50), use_container_width=True)

        buf = io.StringIO()
        scored.to_csv(buf, index=False)
        st.download_button("⬇ Download results CSV", buf.getvalue(),
                           file_name="classified_results.csv", mime="text/csv", type="primary")
