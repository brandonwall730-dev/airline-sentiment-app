import streamlit as st
import pandas as pd
import joblib
import warnings
import os
import re
from PIL import Image

warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AirSentiment · Airline Tweet Analyser",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0f172a 0%, #1e293b 100%);
    color: white;
}
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stRadio [role="radio"][aria-checked="true"] + div { color: #38bdf8 !important; font-weight: 600; }

/* Header */
.hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 60%, #0ea5e9 100%);
    padding: 2.5rem 2rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    color: white;
}
.hero h1 { font-family: 'Space Grotesk', sans-serif; font-size: 2.4rem; margin: 0; font-weight: 700; letter-spacing: -0.5px; }
.hero p { font-size: 1rem; opacity: 0.8; margin: 0.4rem 0 0; }

/* Metric cards */
.metric-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
.metric-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    flex: 1;
    min-width: 140px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.metric-card .label { font-size: 0.72rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600; }
.metric-card .value { font-family: 'Space Grotesk', sans-serif; font-size: 2rem; font-weight: 700; color: #0f172a; line-height: 1.2; }
.metric-card .sub { font-size: 0.8rem; color: #94a3b8; margin-top: 0.1rem; }

/* Sentiment pill */
.pill {
    display: inline-block;
    padding: 0.35em 0.9em;
    border-radius: 50px;
    font-weight: 600;
    font-size: 0.9rem;
    letter-spacing: 0.02em;
}
.pill-negative { background: #fee2e2; color: #b91c1c; }
.pill-neutral  { background: #fef9c3; color: #92400e; }
.pill-positive { background: #dcfce7; color: #15803d; }

/* Predict box */
.predict-result {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.5rem;
    margin-top: 1rem;
}
.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: #0f172a;
    margin: 1.5rem 0 0.8rem;
    padding-bottom: 0.4rem;
    border-bottom: 2px solid #e2e8f0;
}

/* Tab styling */
button[data-baseweb="tab"] { font-weight: 600; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# ── Load assets ────────────────────────────────────────────────────────────────
BASE = os.path.dirname(__file__)
UPLOADS = "."  # adjust if running locally

@st.cache_resource
def load_models():
    nb  = joblib.load(os.path.join(UPLOADS, "nb_model.pkl"))
    svm = joblib.load(os.path.join(UPLOADS, "svm_model.pkl"))
    champ = joblib.load(os.path.join(UPLOADS, "champion_model.pkl"))
    return nb, svm, champ

@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(UPLOADS, "tweets_clean.csv"))
    pred = pd.read_csv(os.path.join(UPLOADS, "predictions_full.csv"))
    metrics = pd.read_csv(os.path.join(UPLOADS, "model_comparison.csv"))
    return df, pred, metrics

def img(fname):
    return Image.open(os.path.join(UPLOADS, fname))

try:
    nb_model, svm_model, champ_model = load_models()
    df, pred_df, metrics_df = load_data()
    MODELS_OK = True
except Exception as e:
    MODELS_OK = False
    MODEL_ERROR = str(e)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ✈️ AirSentiment")
    st.markdown("*US Airline Tweet Analysis*")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Overview", "📊 Exploratory Analysis", "🤖 Model Performance", "🔍 Live Predictor", "📁 Data Explorer"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.78rem; color:#94a3b8; line-height:1.6'>
    <b style='color:#e2e8f0'>Dataset</b><br>
    14,109 tweets · Feb 2015<br>
    6 US Airlines<br><br>
    <b style='color:#e2e8f0'>Models</b><br>
    Naïve Bayes · SVM<br>
    Champion: SVM (79.1% acc)
    </div>
    """, unsafe_allow_html=True)

# ── OVERVIEW ───────────────────────────────────────────────────────────────────
if page == "🏠 Overview":
    st.markdown("""
    <div class="hero">
        <h1>✈️ Airline Sentiment Analysis</h1>
        <p>NLP-powered classification of US airline tweets — Final Year Project</p>
    </div>
    """, unsafe_allow_html=True)

    # Key metrics
    total = len(df)
    neg_pct = round(df["airline_sentiment"].value_counts(normalize=True)["negative"]*100, 1)
    best_acc = round(metrics_df["Accuracy"].max()*100, 1)

    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card">
            <div class="label">Total Tweets</div>
            <div class="value">{total:,}</div>
            <div class="sub">Feb 16 – 24, 2015</div>
        </div>
        <div class="metric-card">
            <div class="label">Negative Rate</div>
            <div class="value">{neg_pct}%</div>
            <div class="sub">Industry average</div>
        </div>
        <div class="metric-card">
            <div class="label">Best Accuracy</div>
            <div class="value">{best_acc}%</div>
            <div class="sub">SVM Champion Model</div>
        </div>
        <div class="metric-card">
            <div class="label">Airlines Covered</div>
            <div class="value">6</div>
            <div class="sub">United · AA · Delta +3</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">Sentiment Distribution</div>', unsafe_allow_html=True)
        st.image(img("class_dist.png"), use_container_width=True)
    with col2:
        st.markdown('<div class="section-title">Net Brand Sentiment Score</div>', unsafe_allow_html=True)
        st.image(img("net_sentiment.png"), use_container_width=True)

    st.markdown('<div class="section-title">Sentiment Over Time (Feb 2015)</div>', unsafe_allow_html=True)
    st.image(img("sentiment_trend.png"), use_container_width=True)

    st.markdown("""
    ---
    **Project Summary:** This project applies Natural Language Processing to classify airline customer tweets
    as *negative*, *neutral*, or *positive*. Two models were trained — Naïve Bayes and a Support Vector Machine (SVM).
    The SVM emerged as the champion model with 79.1% accuracy and a macro F1 of 0.72.
    """)

# ── EXPLORATORY ANALYSIS ───────────────────────────────────────────────────────
elif page == "📊 Exploratory Analysis":
    st.markdown('<div class="hero"><h1>📊 Exploratory Analysis</h1><p>Patterns in customer feedback across 6 US airlines</p></div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["✈️ Per-Airline", "❌ Negative Reasons", "💬 Word Clouds", "🗺️ Brand Perception"])

    with tab1:
        st.markdown('<div class="section-title">Sentiment Breakdown by Airline</div>', unsafe_allow_html=True)
        st.image(img("airline_sentiment.png"), use_container_width=True)

        # mini table
        airline_stats = df.groupby("airline")["airline_sentiment"].value_counts(normalize=True).unstack()*100
        airline_stats = airline_stats.round(1).rename(columns={"negative":"Negative %","neutral":"Neutral %","positive":"Positive %"})
        airline_stats.index.name = "Airline"
        st.dataframe(airline_stats.sort_values("Negative %", ascending=False).style.format("{:.1f}"), use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-title">Top Negative Reasons</div>', unsafe_allow_html=True)
            st.image(img("negative_reason_breakdown.png"), use_container_width=True)
        with col2:
            st.markdown('<div class="section-title">Reasons by Airline</div>', unsafe_allow_html=True)
            st.image(img("neg_reasons_by_airline.png"), use_container_width=True)

        st.markdown("**Confidence in Negative Reason Labels**")
        st.image(img("negativereason_confidence_distribution.png"), use_container_width=True)

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🔴 Negative Tweets")
            st.image(img("wc_negative.png"), use_container_width=True)
        with col2:
            st.markdown("### 🟢 Positive Tweets")
            st.image(img("wc_positive.png"), use_container_width=True)

    with tab4:
        st.image(img("brand_perception_map.png"), use_container_width=True)
        st.markdown("""
        > **Insight:** Virgin America has the lowest negative rate (~39%) while US Airways has the highest (~79%).
        > Delta and Southwest represent the mid-tier, with moderate negativity and meaningful positive engagement.
        """)

# ── MODEL PERFORMANCE ──────────────────────────────────────────────────────────
elif page == "🤖 Model Performance":
    st.markdown('<div class="hero"><h1>🤖 Model Performance</h1><p>Naïve Bayes vs SVM — head-to-head evaluation</p></div>', unsafe_allow_html=True)

    # Metrics table
    st.markdown('<div class="section-title">Performance Metrics</div>', unsafe_allow_html=True)
    display_metrics = metrics_df.copy()
    display_metrics["Accuracy"] = (display_metrics["Accuracy"]*100).round(2).astype(str) + "%"
    display_metrics["Macro F1"] = (display_metrics["Macro F1"]).round(4)
    display_metrics["ROC-AUC"] = (display_metrics["ROC-AUC"]).round(4)
    display_metrics = display_metrics.rename(columns={"Model": "Model", "Macro F1": "Macro F1", "ROC-AUC": "ROC-AUC"})
    st.dataframe(display_metrics.set_index("Model"), use_container_width=True)

    st.markdown("""
    <div style='background:#f0fdf4;border-left:4px solid #22c55e;padding:0.8rem 1rem;border-radius:0 8px 8px 0;margin-bottom:1rem'>
    🏆 <strong>Champion Model: SVM</strong> — outperforms Naïve Bayes on all three metrics, 
    especially Macro F1 (0.72 vs 0.64), reflecting better handling of the imbalanced class distribution.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Confusion Matrices</div>', unsafe_allow_html=True)
    st.image(img("confusion_matrices.png"), use_container_width=True)

    st.markdown("""
    **Reading the matrices:**
    - Naïve Bayes correctly identifies negative tweets well (1,725/1,808) but struggles with positive class (only 201/435 correct).
    - SVM has a more balanced performance, correctly classifying 298 positive and 337 neutral tweets — a significant improvement.
    """)

    st.markdown('<div class="section-title">Prediction Confidence Distribution</div>', unsafe_allow_html=True)
    st.image(img("airline_sentiment_confidence_distribution.png"), use_container_width=True)
    st.markdown("> The bimodal distribution shows the model is either very confident (≥0.95) or moderately confident (~0.65). Low-confidence predictions cluster in the neutral class — the hardest to separate.")

# ── LIVE PREDICTOR ─────────────────────────────────────────────────────────────
elif page == "🔍 Live Predictor":
    st.markdown('<div class="hero"><h1>🔍 Live Tweet Predictor</h1><p>Type any airline tweet and get an instant sentiment prediction</p></div>', unsafe_allow_html=True)

    if not MODELS_OK:
        st.error(f"Could not load models: {MODEL_ERROR}")
    else:
        col1, col2 = st.columns([2, 1])
        with col1:
            model_choice = st.selectbox("Choose model", ["SVM (Champion)", "Naïve Bayes"], index=0)
            tweet_input = st.text_area(
                "Enter a tweet about an airline:",
                placeholder="e.g. @united my flight was cancelled and no one helped me at the gate!",
                height=120
            )

            examples = [
                "Flight delayed 3 hours, no updates, terrible customer service",
                "Amazing crew on my Delta flight today, best experience ever!",
                "My bag is lost and the agent just shrugged",
                "Thanks Southwest for the quick rebooking, really appreciate it",
                "@USAirways I've been on hold for 2 hours",
            ]
            st.caption("Or try an example:")
            ex_cols = st.columns(3)
            for i, ex in enumerate(examples[:3]):
                if ex_cols[i].button(ex[:35]+"…", key=f"ex{i}"):
                    tweet_input = ex
            ex_cols2 = st.columns(2)
            for i, ex in enumerate(examples[3:]):
                if ex_cols2[i].button(ex[:35]+"…", key=f"ex2{i}"):
                    tweet_input = ex

            if st.button("🔮 Predict Sentiment", type="primary", use_container_width=True):
                if tweet_input.strip():
                    model = nb_model if "Naïve" in model_choice else champ_model
                    pred = model.predict([tweet_input])[0]

                    pill_class = f"pill-{pred}"
                    emoji = {"negative": "🔴", "neutral": "🟡", "positive": "🟢"}[pred]

                    result_html = f"""
                    <div class="predict-result">
                        <div style="font-size:0.8rem;color:#64748b;margin-bottom:0.5rem">MODEL: {model_choice.upper()}</div>
                        <div style="font-size:1rem;color:#475569;margin-bottom:1rem;font-style:italic">"{tweet_input}"</div>
                        <div style="font-size:1.5rem;font-weight:700;color:#0f172a;margin-bottom:0.5rem">
                            {emoji} Predicted Sentiment
                        </div>
                        <span class="pill {pill_class}" style="font-size:1.1rem">{pred.upper()}</span>
                    </div>
                    """
                    st.markdown(result_html, unsafe_allow_html=True)

                    if "Naïve" in model_choice:
                        proba = nb_model.predict_proba([tweet_input])[0]
                        classes = nb_model.classes_
                        st.markdown("**Confidence breakdown:**")
                        for cls, prob in zip(classes, proba):
                            color = {"negative":"#ef4444","neutral":"#f59e0b","positive":"#22c55e"}[cls]
                            st.markdown(f"`{cls}`: {prob*100:.1f}%")
                            st.progress(float(prob))
                else:
                    st.warning("Please enter a tweet first.")

        with col2:
            st.markdown("### 💡 Tips")
            st.info("""
**For better predictions:**
- Include airline-related words (flight, gate, bag, crew)
- Mention specific issues (delay, cancel, lost)
- Express emotion naturally
- Tweets can be typed or pasted

**The SVM model is best at:**
- Detecting negative sentiment
- Handling imbalanced classes
- Short, punchy tweets

**Naïve Bayes gives:**
- Confidence probabilities
- Faster predictions
            """)

# ── DATA EXPLORER ──────────────────────────────────────────────────────────────
elif page == "📁 Data Explorer":
    st.markdown('<div class="hero"><h1>📁 Data Explorer</h1><p>Browse and filter the full prediction dataset</p></div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        airline_filter = st.multiselect("Airline", sorted(df["airline"].unique()), default=list(df["airline"].unique()))
    with col2:
        sentiment_filter = st.multiselect("Sentiment", ["negative", "neutral", "positive"], default=["negative", "neutral", "positive"])
    with col3:
        reason_opts = ["All"] + sorted(df["negativereason"].dropna().unique().tolist())
        reason_filter = st.selectbox("Negative Reason", reason_opts)

    filtered = df[df["airline"].isin(airline_filter) & df["airline_sentiment"].isin(sentiment_filter)]
    if reason_filter != "All":
        filtered = filtered[filtered["negativereason"] == reason_filter]

    st.markdown(f"**{len(filtered):,} tweets** matching your filters")

    show_df = filtered[["airline", "airline_sentiment", "negativereason", "text_final"]].rename(columns={
        "airline": "Airline",
        "airline_sentiment": "Sentiment",
        "negativereason": "Negative Reason",
        "text_final": "Tweet Text"
    }).reset_index(drop=True)

    st.dataframe(
        show_df.head(200),
        use_container_width=True,
        height=450,
        column_config={
            "Sentiment": st.column_config.TextColumn(width="small"),
            "Airline": st.column_config.TextColumn(width="medium"),
        }
    )

    if len(filtered) > 200:
        st.caption(f"Showing first 200 of {len(filtered):,} results")
