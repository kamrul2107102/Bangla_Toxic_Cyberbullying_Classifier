from pathlib import Path
import json
import time
import streamlit as st
import pandas as pd
import numpy as np

from src.config import ARTIFACT_DIR, LABELS
from src.preprocessing import preprocess
from src.predict import load_model

st.set_page_config(
    page_title="Bangla Toxic & Cyberbullying Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling for clean, professional aesthetics and EQUAL HEIGHT CARDS
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-caption {
        font-size: 1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .benchmark-notice {
        background-color: #FEF3C7;
        border-left: 5px solid #F59E0B;
        padding: 0.8rem 1.2rem;
        border-radius: 4px;
        color: #92400E;
        font-weight: 500;
        margin-bottom: 1rem;
    }

    /* EQUAL HEIGHT UNIFIED MODEL CARDS */
    .unified-model-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);
        display: flex;
        flex-direction: column;
        height: 100%;
        min-height: 450px;
        box-sizing: border-box;
        margin-bottom: 1rem;
    }
    .card-header-section {
        min-height: 72px;
        margin-bottom: 0.6rem;
    }
    .card-title {
        font-size: 1.08rem;
        font-weight: 700;
        color: #0F172A;
        line-height: 1.35;
        margin-bottom: 0.35rem;
    }
    .card-badge {
        display: inline-block;
        font-size: 0.74rem;
        background: #F1F5F9;
        color: #475569;
        padding: 2px 7px;
        border-radius: 4px;
        font-weight: 600;
    }
    .card-verdict-section {
        min-height: 44px;
        display: flex;
        align-items: center;
        padding: 8px 12px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.95rem;
        margin-bottom: 0.75rem;
    }
    .card-verdict-toxic {
        background-color: #FEE2E2;
        color: #991B1B;
        border: 1px solid #FCA5A5;
    }
    .card-verdict-safe {
        background-color: #DCFCE7;
        color: #166534;
        border: 1px solid #86EFAC;
    }
    .card-labels-section {
        min-height: 68px;
        margin-bottom: 0.75rem;
    }
    .card-labels-title {
        font-size: 0.78rem;
        font-weight: 600;
        color: #64748B;
        margin-bottom: 5px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .card-labels-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 4px;
        align-items: flex-start;
    }
    .label-pill {
        background: #EFF6FF;
        color: #1D4ED8;
        border: 1px solid #BFDBFE;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 2px 6px;
        border-radius: 4px;
    }
    .label-pill-none {
        color: #94A3B8;
        font-style: italic;
        font-size: 0.8rem;
    }
    .card-latency-section {
        font-size: 0.82rem;
        color: #64748B;
        margin-bottom: 0.75rem;
        padding-bottom: 0.6rem;
        border-bottom: 1px solid #F1F5F9;
    }
    .card-probs-section {
        flex-grow: 1;
    }
    .card-probs-title {
        font-size: 0.78rem;
        font-weight: 600;
        color: #64748B;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .prob-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
        font-size: 0.83rem;
    }
    .prob-bar-container {
        width: 50%;
        background: #F1F5F9;
        border-radius: 4px;
        height: 7px;
        overflow: hidden;
    }
    .prob-bar-fill {
        height: 100%;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🛡️ Bangla Toxic Comment & Cyberbullying Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-caption">Multi-Label Toxicity, Threat & Cyberbullying Detection for Bengali and Banglish — Multi-Model Benchmark Arena</div>', unsafe_allow_html=True)

# All supported models definitions
ALL_MODELS = {
    "naive_bayes": {
        "name": "Naive Bayes",
        "method": "Closed-form Bayes + Laplace Smoothing",
        "type": "Classical Generative"
    },
    "tfidf_lr": {
        "name": "TF-IDF + Logistic Regression",
        "method": "Sparse TF-IDF n-grams + Sigmoid LogReg",
        "type": "Classical Discriminative"
    },
    "word2vec_lr": {
        "name": "Word2Vec + Logistic Regression",
        "method": "Skip-Gram SGNS + Mean Embedding Pooling",
        "type": "Dense Embedding + LogReg"
    },
    "bilstm": {
        "name": "BiLSTM",
        "method": "PyTorch Stacked BiLSTM + Trainable Embeddings",
        "type": "Recurrent Sequence Model"
    },
    "transformer": {
        "name": "Transformer Encoder",
        "method": "PyTorch Self-Attention + Sinusoidal Positional Encoding",
        "type": "Transformer Sequence Model"
    },
    "banglabert": {
        "name": "BanglaBERT",
        "method": "Pretrained ELECTRA Discriminator (CSE BUET)",
        "type": "Pretrained Foundation Model"
    }
}

# Cache model loader in memory for instant multi-model inference
@st.cache_resource
def get_cached_model(m_name: str):
    return load_model(m_name)

# Helper to check which models are trained and available on disk
def get_available_models():
    available = {}
    for m_id, m_info in ALL_MODELS.items():
        joblib_f = ARTIFACT_DIR / f"{m_id}.joblib"
        dir_f = ARTIFACT_DIR / m_id
        if joblib_f.exists() or (dir_f.is_dir() and (dir_f / "model.pt").exists()):
            available[m_id] = m_info
    return available

available_models = get_available_models()

def load_thresholds(m_name: str):
    metrics_path = ARTIFACT_DIR / f"{m_name}_metrics.json"
    th = {lbl: 0.5 for lbl in LABELS}
    if metrics_path.exists():
        try:
            data = json.loads(metrics_path.read_text(encoding="utf-8"))
            raw = data.get("best_thresholds", {})
            if isinstance(raw, dict):
                for lbl in LABELS:
                    if lbl in raw:
                        th[lbl] = float(raw[lbl])
        except Exception:
            pass
    return th

MODE_COMPARE = "⚔️ Unified Multi-Model Comparison (Compare All Models)"
MODE_SINGLE = "🔬 Single Model Deep-Dive"

single_model_name = list(available_models.keys())[0] if available_models else "naive_bayes"
active_single_th = load_thresholds(single_model_name)

with st.sidebar:
    st.header("⚙️ Dashboard Mode")
    view_mode = st.radio(
        "Select display view:",
        [MODE_COMPARE, MODE_SINGLE],
        index=0
    )
    
    st.markdown("---")
    
    if view_mode == MODE_SINGLE:
        st.subheader("Model Settings")
        model_options = {k: v["name"] for k, v in ALL_MODELS.items()}
        selected_label = st.selectbox(
            "Select active model:",
            options=list(model_options.values()),
            index=0
        )
        single_model_name = [k for k, v in model_options.items() if v == selected_label][0]
        
        if single_model_name == "banglabert":
            st.markdown(
                '<div class="benchmark-notice"><b>BanglaBERT</b> is an external pretrained benchmark (ELECTRA) and is not part of the from-scratch model pipeline.</div>',
                unsafe_allow_html=True
            )
            
        single_th_dict = load_thresholds(single_model_name)
        th_mode = st.radio(
            "Threshold Mode:",
            ["🎯 Optimal Validation Thresholds (Recommended)", "🎚️ Custom Threshold Slider"]
        )
        if th_mode == "🎚️ Custom Threshold Slider":
            c_th = float(st.slider("Detection Threshold:", 0.05, 0.95, 0.50, 0.05))
            active_single_th = {lbl: c_th for lbl in LABELS}
        else:
            active_single_th = single_th_dict
            
        st.markdown("---")
        st.subheader("Model Performance")
        metrics_path = ARTIFACT_DIR / f"{single_model_name}_metrics.json"
        if metrics_path.exists():
            try:
                rep = json.loads(metrics_path.read_text(encoding="utf-8"))
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    st.metric("Test Micro-F1", f"{rep.get('test', {}).get('micro_f1', 0.0):.3f}")
                with col_m2:
                    st.metric("Test Macro-F1", f"{rep.get('test', {}).get('macro_f1', 0.0):.3f}")
                with st.expander("Active Thresholds"):
                    for lbl, th in active_single_th.items():
                        disp_lbl = "Gender Discrimination" if lbl == "sexist" else lbl.replace("_", " ").title()
                        st.write(f"• **{disp_lbl}**: `{th:.2f}` ({th*100:.0f}%)")
            except Exception:
                pass
    else:
        st.subheader("🎯 Active Model Fleet (Ready to Compare)")
        st.caption(f"Currently {len(available_models)} trained models available:")
        for mid, info in available_models.items():
            st.markdown(f"• **{info['name']}**  \n  *{info['method']}*")
            
        st.markdown("---")
        st.info("ℹ️ **Comparison Arena:** Enter a comment to evaluate all trained models simultaneously, side-by-side with consensus metrics.")

    st.markdown("---")
    st.caption("NLP Multi-Label Toxic Classifier • Classical to Neural")

# Example Sentences for Quick Testing
SAMPLE_SENTENCES = {
    "✨ Select a preset sample...": "",
    "🔴 [Threat] তোরে যেখানে পামু মাইরা তক্তা বানায়া ফেলমু, জানে শেষ করে দেব": "তোরে যেখানে পামু মাইরা তক্তা বানায়া ফেলমু, জানে শেষ করে দেব",
    "🟢 [Neutral Question] এই বইটা কোথায় কিনতে পাওয়া যাবে?": "এই বইটা কোথায় কিনতে পাওয়া যাবে?",
    "🟢 [Positive / Neutral] আজকের আবহাওয়াটা অনেক সুন্দর এবং চমৎকার।": "আজকের আবহাওয়াটা অনেক সুন্দর এবং চমৎকার।",
    "🟢 [Praise] ভাই আপনার কাজটি সত্যিই অনেক তথ্যবহুল ও দারুণ ছিল!": "ভাই আপনার কাজটি সত্যিই অনেক তথ্যবহুল ও দারুণ ছিল!",
    "🔴 [Bullying / Abusive] তুই একটা আস্ত কুত্তার বাচ্চা তোরে জুতা মারমু": "তুই একটা আস্ত কুত্তার বাচ্চা তোরে জুতা মারমু",
    "🔴 [Hate Speech] এই ফকিন্নি মালাউনদের দেশ থেকে লাথি মেরে তাড়ানো উচিত": "এই ফকিন্নি মালাউনদের দেশ থেকে লাথি মেরে তাড়ানো উচিত",
    "🟡 [Banglish Positive] video ta onek sundor hoyeche bro, shuvokamona roilo": "video ta onek sundor hoyeche bro, shuvokamona roilo",
    "🔴 [Banglish Toxic] tui ekta baje faltu manush, tore dekhle shobai ghenna kore": "tui ekta baje faltu manush, tore dekhle shobai ghenna kore"
}

if "user_comment" not in st.session_state:
    st.session_state.user_comment = "তোরে যেখানে পামু মাইরা তক্তা বানায়া ফেলমু, জানে শেষ করে দেব"

def on_sample_change():
    chosen = st.session_state.sample_selector
    if chosen and SAMPLE_SENTENCES.get(chosen):
        st.session_state.user_comment = SAMPLE_SENTENCES[chosen]

st.markdown("##### 🧪 Sample Test Sentences:")
st.selectbox(
    "Choose a preset test sample:",
    options=list(SAMPLE_SENTENCES.keys()),
    index=0,
    key="sample_selector",
    on_change=on_sample_change,
    label_visibility="collapsed"
)

# Text area
text = st.text_area(
    "Enter Bengali or Banglish comment to analyze:",
    key="user_comment",
    height=110,
    placeholder="Type or paste your text here..."
)

col_btn, _ = st.columns([1, 3])
with col_btn:
    classify_clicked = st.button("🔍 Classify & Compare", type="primary", width="stretch")

label_display_map = {
    "hate_speech": "Hate Speech (ঘৃণামূলক বক্তব্য)",
    "sexist": "Gender Discrimination (লিঙ্গবৈষম্য)",
    "threat": "Threat / Violence (হুমকি)",
    "bullying": "Bullying (সাইবারবুলিং)",
    "toxic": "Toxic / Abusive (ক্ষতিকর/অশালীন)"
}

label_badge_map = {
    "hate_speech": "HATE SPEECH",
    "sexist": "GENDER DISCRIMINATION",
    "threat": "THREAT",
    "bullying": "BULLYING",
    "toxic": "TOXIC"
}

label_short_map = {
    "hate_speech": "Hate Speech",
    "sexist": "Gender Discrimination",
    "threat": "Threat",
    "bullying": "Bullying",
    "toxic": "Toxic"
}

def run_single_inference(m_name: str, comment_text: str, th_dict: dict):
    m = get_cached_model(m_name)
    start_t = time.perf_counter()
    probs = np.asarray(m.predict_proba([comment_text]))[0]
    latency_ms = (time.perf_counter() - start_t) * 1000.0
    detected = []
    for i, lbl in enumerate(LABELS):
        if probs[i] >= float(th_dict.get(lbl, 0.5)):
            detected.append(lbl)
    return {
        "model_name": m_name,
        "probs": {LABELS[i]: float(probs[i]) for i in range(len(LABELS))},
        "detected": detected,
        "is_neutral": len(detected) == 0,
        "latency_ms": latency_ms
    }

if classify_clicked or text.strip():
    if not text.strip():
        st.warning("⚠️ Please enter a text or select a preset sample.")
    else:
        st.markdown("---")
        
        # =========================================================================
        # 1. UNIFIED MULTI-MODEL COMPARISON VIEW
        # =========================================================================
        if view_mode == MODE_COMPARE:
            if not available_models:
                st.error("No trained model artifacts found on disk.")
            else:
                results = {}
                with st.spinner("Running inference across all ready models..."):
                    for mid in available_models.keys():
                        th_d = load_thresholds(mid)
                        try:
                            results[mid] = run_single_inference(mid, text, th_d)
                        except Exception as e:
                            st.warning(f"Error executing {mid}: {e}")
                
                # Consensus calculation
                total_evaluated = len(results)
                toxic_votes = sum(1 for r in results.values() if not r["is_neutral"])
                neutral_votes = total_evaluated - toxic_votes
                
                # --- Consensus Banner ---
                st.markdown("### 🏆 Model Consensus & Agreement")
                if toxic_votes == total_evaluated:
                    st.error(f"🚨 **Unanimous Toxic Agreement ({total_evaluated}/{total_evaluated} Models Agree):** The input is classified as harmful/toxic.")
                elif neutral_votes == total_evaluated:
                    st.success(f"✅ **Unanimous Clean Agreement ({total_evaluated}/{total_evaluated} Models Agree):** The input is classified as safe and neutral.")
                else:
                    st.warning(f"⚖️ **Split Verdict ({toxic_votes}/{total_evaluated} Models Flagged Toxic):** Differing classification thresholds among models.")
                
                # --- Side-by-Side Model Cards with EQUAL HEIGHTS ---
                st.markdown("### 📊 Side-by-Side Model Verdicts")
                model_cols = st.columns(total_evaluated)
                for idx, (mid, res) in enumerate(results.items()):
                    with model_cols[idx]:
                        m_meta = ALL_MODELS.get(mid, {})
                        
                        # Build detected label pills HTML
                        if res["detected"]:
                            badges_html = " ".join([f"<span class='label-pill'>{label_badge_map.get(l, l.upper())}</span>" for l in res["detected"]])
                        else:
                            badges_html = "<span class='label-pill-none'>None (Safe / Normal)</span>"
                            
                        # Verdict class and text
                        if res["is_neutral"]:
                            verdict_class = "card-verdict-safe"
                            verdict_html = "✅ Safe (Clean / Neutral)"
                        else:
                            verdict_class = "card-verdict-toxic"
                            verdict_html = "🚨 Toxic (Harmful Content)"
                            
                        # Build top 3 probability bars
                        top_sorted = sorted(res["probs"].items(), key=lambda x: x[1], reverse=True)[:3]
                        prob_rows_html = []
                        for lbl, p in top_sorted:
                            pct = p * 100.0
                            bar_color = "#EF4444" if p >= 0.5 else "#3B82F6"
                            disp_name = label_short_map.get(lbl, lbl)
                            prob_rows_html.append(
                                f'<div class="prob-row">'
                                f'<span style="font-weight: 500; color: #334155;">{disp_name}:</span>'
                                f'<div class="prob-bar-container">'
                                f'<div class="prob-bar-fill" style="width: {pct:.1f}%; background-color: {bar_color};"></div>'
                                f'</div>'
                                f'<span style="font-weight: 600; color: #0F172A; width: 45px; text-align: right;">{pct:.1f}%</span>'
                                f'</div>'
                            )
                        probs_html = "".join(prob_rows_html)

                        # Render complete unified card with standardized sections (unindented to prevent markdown code block)
                        card_html = (
                            f'<div class="unified-model-card">'
                            f'<div class="card-header-section">'
                            f'<div class="card-title">{m_meta.get("name", mid)}</div>'
                            f'<span class="card-badge">{m_meta.get("type", "Model")}</span>'
                            f'</div>'
                            f'<div class="card-verdict-section {verdict_class}">{verdict_html}</div>'
                            f'<div class="card-labels-section">'
                            f'<div class="card-labels-title">Detected Labels:</div>'
                            f'<div class="card-labels-badges">{badges_html}</div>'
                            f'</div>'
                            f'<div class="card-latency-section">⏱️ Inference Latency: <b>{res["latency_ms"]:.2f} ms</b></div>'
                            f'<div class="card-probs-section">'
                            f'<div class="card-probs-title">Top Predicted Confidences:</div>'
                            f'{probs_html}'
                            f'</div>'
                            f'</div>'
                        )
                        st.markdown(card_html, unsafe_allow_html=True)
                
                # --- Unified Probability Matrix Table ---
                st.markdown("### 📈 Unified Multi-Model Probability Matrix")
                st.caption("Side-by-side comparison of predicted confidence probabilities (%) for all 5 canonical toxicity labels:")
                
                table_rows = []
                for lbl in LABELS:
                    row_data = {"Toxicity Label": label_display_map.get(lbl, lbl)}
                    model_probs = []
                    any_detected = False
                    for mid in results.keys():
                        prob = results[mid]["probs"][lbl]
                        model_probs.append(prob)
                        th = float(load_thresholds(mid).get(lbl, 0.5))
                        is_flagged = prob >= th
                        if is_flagged:
                            any_detected = True
                        flag_icon = "🚨 " if is_flagged else ""
                        row_data[ALL_MODELS[mid]["name"].split("(")[0].strip()] = f"{flag_icon}{prob*100:.1f}%"
                    
                    avg_prob = np.mean(model_probs) if model_probs else 0.0
                    row_data["Ensemble Avg (%)"] = f"{avg_prob*100:.1f}%"
                    row_data["Consensus Status"] = "🚨 DETECTED" if any_detected else "✅ Clean"
                    table_rows.append(row_data)
                    
                df_matrix = pd.DataFrame(table_rows)
                st.dataframe(df_matrix, hide_index=True, width="stretch")
        
        # =========================================================================
        # 2. SINGLE MODEL DEEP-DIVE VIEW
        # =========================================================================
        else:
            try:
                single_res = run_single_inference(single_model_name, text, active_single_th)
                st.markdown(f"### 🔬 Detailed Analysis: {ALL_MODELS[single_model_name]['name']}")
                
                if single_res["is_neutral"]:
                    st.success("✅ **Clean & Safe (Neutral)** — No toxicity thresholds exceeded.")
                else:
                    detected_badges = " ".join([f"`{label_badge_map.get(lbl, lbl.upper())}`" for lbl in single_res["detected"]])
                    st.error(f"🚨 **Harmful Content Detected:** {detected_badges}")
                
                rows = []
                for lbl in LABELS:
                    prob = single_res["probs"][lbl]
                    th = float(active_single_th.get(lbl, 0.5))
                    is_det = prob >= th
                    rows.append({
                        "Label": label_display_map.get(lbl, lbl),
                        "Percentage (%)": f"{prob*100:.1f}%",
                        "Probability (0-1)": round(prob, 4),
                        "Cutoff Threshold": f"{th*100:.0f}%",
                        "Confidence Meter": prob,
                        "Status": "🚨 DETECTED" if is_det else "✅ Clean"
                    })
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            except Exception as ex:
                st.error(f"Error executing {single_model_name}: {ex}")

        # =========================================================================
        # 3. COMMON PREPROCESSED TOKENS INSPECTION
        # =========================================================================
        with st.expander("🔎 Inspected Preprocessed Tokens"):
            st.caption("Tokens after Unicode NFKC normalization, regex cleaning, and dialect mapping:")
            tokens = preprocess(text)
            if tokens:
                st.write(" | ".join([f"`{t}`" for t in tokens]))
            else:
                st.write("*(No significant tokens found)*")


