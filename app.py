from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ============================================================
# Page setup
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "paddydataset.csv"
BUNDLE_PATH = BASE_DIR / "paddy_dashboard_bundle.joblib"

st.set_page_config(
    page_title="Paddy Yield Prediction",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Screenshot-inspired styling
# ============================================================
st.markdown(
    """
    <style>
    :root {
        --accent: #ff3b3f;
        --ink: #252a35;
        --muted: #7c8796;
        --line: #dfe4ea;
        --sidebar: #f2f4f7;
        --hero1: #102e36;
        --hero2: #2d5b68;
        --green: #d9f0df;
    }

    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
    }

    .stApp { background: #ffffff; color: #252a35; }

    /* Remove Streamlit's floating black header/toolbar so it does not cover navigation */
    header[data-testid="stHeader"] {
        height: 0 !important;
        min-height: 0 !important;
        background: transparent !important;
        box-shadow: none !important;
    }
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    [data-testid="stStatusWidget"] {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
    }

    /* Force controls to look clickable even when the browser/system is in dark mode */
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
        background: #ffffff !important;
        border: 1px solid #cbd5df !important;
        border-radius: 8px !important;
        color: #1f2937 !important;
        min-height: 42px !important;
        box-shadow: 0 1px 2px rgba(16,24,40,.04) !important;
    }
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div:hover {
        border-color: #ff4b4b !important;
        box-shadow: 0 0 0 2px rgba(255,75,75,.10) !important;
    }
    section[data-testid="stSidebar"] div[data-baseweb="select"] * {
        color: #1f2937 !important;
    }
    section[data-testid="stSidebar"] input {
        background: #ffffff !important;
        color: #1f2937 !important;
        border-color: #cbd5df !important;
    }
    section[data-testid="stSidebar"] .stButton > button {
        background: #ff4b4b !important;
        color: #ffffff !important;
        border: 1px solid #ff4b4b !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        min-height: 44px !important;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: #e63f43 !important;
        border-color: #e63f43 !important;
    }

    /* Tabs: stronger active state, similar to the reference screenshot */
    button[data-baseweb="tab"] {
        border-radius: 6px 6px 0 0 !important;
    }
    button[data-baseweb="tab"]:hover {
        background: #fff4f4 !important;
        color: #e5393e !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background: #fff7f7 !important;
        font-weight: 700 !important;
    }

    /* Keep tables light/readable regardless of Streamlit theme */
    div[data-testid="stDataFrame"] {
        background: #ffffff !important;
        color: #1f2937 !important;
    }

    /* Main area dimensions, close to the reference screenshots */
    .block-container {
        max-width: 1180px;
        padding-top: 0.55rem;
        padding-bottom: 4rem;
        padding-left: 2.25rem;
        padding-right: 2.25rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: var(--sidebar);
        border-right: 1px solid #e1e5e9;
        width: 275px !important;
        min-width: 275px !important;
    }
    section[data-testid="stSidebar"] > div:first-child {
        width: 275px !important;
        padding-top: 1.6rem;
    }
    section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.62rem;
    }
    section[data-testid="stSidebar"] hr {
        margin: 1.25rem 0;
        border-color: #d5dae0;
    }
    section[data-testid="stSidebar"] label p {
        font-size: 0.77rem !important;
        color: #343b48 !important;
    }
    section[data-testid="stSidebar"] h3 {
        font-size: 0.88rem;
        margin-top: 0.2rem;
        margin-bottom: 0.15rem;
        color: #313846;
    }

    .sidebar-title {
        text-align: center;
        padding: 2.8rem 0.65rem 2.0rem;
    }
    .sidebar-title h2 {
        margin: 0;
        color: #343a46;
        font-size: 1.28rem;
        line-height: 1.18;
        font-weight: 750;
    }
    .sidebar-title p {
        margin: 1.2rem 0 0;
        color: #7c8792;
        font-size: 0.72rem;
        letter-spacing: 0.02em;
    }

    /* Top navigation tabs */
    div[data-testid="stTabs"] > div:first-child {
        gap: 0.25rem;
        border-bottom: 1px solid #dde2e8;
    }
    button[data-baseweb="tab"] {
        padding: 0.55rem 0.72rem 0.72rem !important;
        font-size: 0.79rem !important;
        color: #2f3540 !important;
        white-space: nowrap;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: var(--accent) !important;
    }
    [data-baseweb="tab-highlight"] {
        background-color: var(--accent) !important;
        height: 2px !important;
    }

    /* Nested tabs */
    .stTabs .stTabs button[data-baseweb="tab"] {
        font-size: 0.77rem !important;
        padding-left: 0.55rem !important;
        padding-right: 0.55rem !important;
    }

    h1, h2, h3, h4 { color: var(--ink); letter-spacing: -0.01em; }
    h2 { font-size: 1.18rem; }
    h3 { font-size: 1.02rem; }

    .hero {
        background: linear-gradient(112deg, var(--hero1) 0%, #1e4652 53%, var(--hero2) 100%);
        border-radius: 16px;
        padding: 34px 36px;
        margin: 0.85rem 0 1.5rem;
        box-shadow: 0 14px 28px rgba(20, 47, 55, 0.18);
    }
    .hero h1 {
        margin: 0;
        color: white !important;
        font-size: 2.0rem;
        font-weight: 760;
    }
    .hero p {
        margin: 0.7rem 0 0;
        color: #e1ecef;
        font-size: 0.99rem;
    }

    .section-title {
        font-size: 1.03rem;
        font-weight: 750;
        color: #2c3340;
        margin: 1.2rem 0 0.35rem;
        padding-bottom: 0.55rem;
        border-bottom: 1px solid #dfe4e9;
    }
    .subtle {
        color: #788391;
        font-size: 0.82rem;
        margin-top: 0.15rem;
    }
    .champion-line {
        margin: 0.7rem 0;
        color: #303643;
        font-size: 0.91rem;
    }

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e3e7ec;
        border-radius: 10px;
        padding: 16px 18px;
        box-shadow: none;
    }
    div[data-testid="stMetricLabel"] p { color: #717b89; font-size: 0.76rem; }
    div[data-testid="stMetricValue"] { color: #222936; }

    div[data-testid="stDataFrame"] {
        border: 1px solid #e0e5ea;
        border-radius: 10px;
        overflow: hidden;
    }

    .prediction-panel {
        border: 1px solid #dce4e8;
        border-radius: 14px;
        background: #f8fafb;
        padding: 22px 24px;
    }
    .prediction-label { color: #7b8592; font-size: .82rem; }
    .prediction-value { color: #1d2733; font-size: 2.05rem; font-weight: 780; margin-top: .2rem; }
    .pill {
        display:inline-block;
        border-radius: 999px;
        padding: 5px 11px;
        background:#e7eef1;
        color:#30434b;
        font-size:.78rem;
        font-weight:700;
    }

    /* keep plot spacing similar to reference */
    div[data-testid="stPlotlyChart"] { margin-top: 0.15rem; }

    /* hide empty spacer lines from markdown */
    .element-container:has(.stMarkdown p:empty) { display:none; }


    /* Reference-like red sliders / controls */
    div[data-baseweb="slider"] div[role="slider"] { background-color: #ff3b3f !important; }
    div[data-baseweb="slider"] > div > div { color: #ff3b3f !important; }
    /* Keep comparison tables visually prominent */
    div[data-testid="stDataFrame"] { background:#fff; }

    @media (max-width: 900px) {
        .block-container { padding-left: 1rem; padding-right: 1rem; }
        section[data-testid="stSidebar"] { min-width: 250px !important; width:250px !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Load data/model artifacts
# ============================================================
@st.cache_resource
def load_bundle():
    return joblib.load(BUNDLE_PATH)


@st.cache_data
def load_data():
    raw = pd.read_csv(DATA_PATH)
    raw.columns = raw.columns.str.strip()
    y = raw["Paddy yield(in Kg)"]
    raw["Paddy Yield Category"] = pd.Categorical(
        np.select(
            [y <= 10000, y <= 20000, y <= 30000],
            ["Low", "Moderate", "High"],
            default="Very High",
        ),
        categories=["Low", "Moderate", "High", "Very High"],
        ordered=True,
    )
    clean = raw.drop_duplicates(
        subset=[c for c in raw.columns if c != "Paddy Yield Category"]
    ).reset_index(drop=True)
    return raw, clean


try:
    bundle = load_bundle()
    raw_df, df = load_data()
except Exception as exc:
    st.error(
        "Dashboard files could not be loaded. Keep app.py, paddydataset.csv, "
        "and paddy_dashboard_bundle.joblib in the same folder."
    )
    st.exception(exc)
    st.stop()

models = bundle["models"]
selected_features = bundle["selected_features"]
num_cols = bundle["numeric_cols"]
cat_cols = bundle["categorical_cols"]
class_labels = [str(x) for x in bundle["class_labels"]]
metrics_df = pd.DataFrame(bundle["metrics"])
cv_folds = bundle["cv_folds"]

# Evaluation comparison exactly as written in the supplied ABCCC (2).ipynb
NOTEBOOK_COMPARISON = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest", "KNN", "ANN"],
    "Accuracy": [97.44, 95.38, 95.21, 89.74],
    "Macro Precision": [98.30, 96.21, 96.18, 94.36],
    "Macro Recall": [97.45, 94.21, 93.32, 89.80],
    "Macro F1-score": [97.78, 95.14, 95.49, 90.41],
})

MODEL_SHORT = {
    "Logistic Regression (Baseline)": "Logistic Regression",
    "KNN (Tuned)": "KNN",
    "Random Forest (Tuned)": "Random Forest",
    "ANN (Tuned)": "ANN",
}
MODEL_ORDER = [m for m in [
    "Logistic Regression (Baseline)",
    "KNN (Tuned)",
    "Random Forest (Tuned)",
    "ANN (Tuned)",
] if m in models]

PLOT_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}


def clean_plot_layout(fig, height=430, margin=None, legend=True):
    if margin is None:
        margin = dict(l=45, r=25, t=65, b=55)
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=margin,
        font=dict(family="Arial, sans-serif", size=12, color="#343a46"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=legend,
        legend=dict(orientation="h", yanchor="top", y=-0.16, xanchor="center", x=0.5),
        hoverlabel=dict(font_size=12),
    )
    fig.update_xaxes(gridcolor="#e3e7eb", zeroline=False)
    fig.update_yaxes(gridcolor="#e3e7eb", zeroline=False)
    return fig


# ============================================================
# Sidebar — reference-like title + sectioned inputs
# ============================================================
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-title">
            <h2>Paddy Yield<br>Prediction</h2>
            <p>BMDS2003 Data Science</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown("### Prediction Model")
    selected_model_name = st.selectbox(
        "Model",
        MODEL_ORDER,
        index=MODEL_ORDER.index("Random Forest (Tuned)") if "Random Forest (Tuned)" in MODEL_ORDER else 0,
        format_func=lambda x: MODEL_SHORT.get(x, x),
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("### Farm Profile")

    user_values = {}
    for col in ["Agriblock", "Variety", "Soil Types", "Nursery"]:
        if col in selected_features:
            user_values[col] = st.selectbox(
                col,
                bundle["categorical_options"][col],
                key=f"side_{col}",
            )

    st.divider()
    st.markdown("### Rainfall & Moisture")

    def numeric_slider(col):
        s = bundle["numeric_stats"][col]
        lo = float(s["min"])
        hi = float(s["max"])
        med = float(s["median"])
        span = hi - lo
        # sensible slider step, keeping decimals where needed
        if span <= 5:
            step = 0.01
        elif span <= 50:
            step = 0.1
        else:
            step = 1.0
        # Streamlit requires the value to stay on range; round for cleaner display
        if step >= 1:
            med = float(round(med))
        else:
            med = round(med, 2)
        return st.slider(
            col,
            min_value=lo,
            max_value=hi,
            value=min(max(med, lo), hi),
            step=step,
            key=f"side_{col}",
        )

    rain_cols = [
        "30DRain( in mm)", "30DAI(in mm)", "30_50DRain( in mm)", "30_50DAI(in mm)",
        "51_70DRain(in mm)", "51_70AI(in mm)", "71_105DRain(in mm)", "71_105DAI(in mm)",
    ]
    for col in rain_cols:
        if col in selected_features:
            user_values[col] = numeric_slider(col)

    st.divider()
    st.markdown("### Temperature")
    temp_cols = [
        "Min temp_D1_D30", "Max temp_D1_D30", "Min temp_D31_D60", "Max temp_D31_D60",
        "Min temp_D61_D90", "Max temp_D61_D90", "Min temp_D91_D120", "Max temp_D91_D120",
    ]
    for col in temp_cols:
        if col in selected_features:
            user_values[col] = numeric_slider(col)

    st.divider()
    st.markdown("### Wind & Humidity")
    wind_speed_cols = [
        "Inst Wind Speed_D1_D30(in Knots)", "Inst Wind Speed_D31_D60(in Knots)",
        "Inst Wind Speed_D61_D90(in Knots)", "Inst Wind Speed_D91_D120(in Knots)",
    ]
    for col in wind_speed_cols:
        if col in selected_features:
            user_values[col] = numeric_slider(col)

    wind_dir_cols = [
        "Wind Direction_D1_D30", "Wind Direction_D31_D60",
        "Wind Direction_D61_D90", "Wind Direction_D91_D120",
    ]
    for col in wind_dir_cols:
        if col in selected_features:
            user_values[col] = st.selectbox(
                col,
                bundle["categorical_options"][col],
                key=f"side_{col}",
            )

    humidity_cols = [
        "Relative Humidity_D1_D30", "Relative Humidity_D31_D60",
        "Relative Humidity_D61_D90", "Relative Humidity_D91_D120",
    ]
    for col in humidity_cols:
        if col in selected_features:
            user_values[col] = numeric_slider(col)

    st.divider()
    st.markdown("### Production Indicator")
    if "Trash(in bundles)" in selected_features:
        user_values["Trash(in bundles)"] = numeric_slider("Trash(in bundles)")

# guard against any feature not covered by the sectioned widgets
for col in selected_features:
    if col not in user_values:
        if col in num_cols:
            user_values[col] = float(bundle["numeric_stats"][col]["median"])
        else:
            user_values[col] = bundle["categorical_options"][col][0]

input_df = pd.DataFrame([{col: user_values[col] for col in selected_features}])
selected_model = models[selected_model_name]
prediction = str(selected_model.predict(input_df)[0])
proba = selected_model.predict_proba(input_df)[0]
model_classes = [str(c) for c in selected_model.named_steps["model"].classes_]
prob_map = {c: float(p) for c, p in zip(model_classes, proba)}
confidence = prob_map[prediction]

# ============================================================
# Main navigation — same overall hierarchy as reference
# ============================================================
main_tabs = st.tabs([
    "Prediction",
    "Data Understanding",
    "Model Performance",
    "Data Preparation",
    "About",
])

# ============================================================
# 1. Prediction
# ============================================================
with main_tabs[0]:
    st.markdown(
        """
        <div class="hero">
            <h1>Paddy Yield Prediction</h1>
            <p>Predict the expected total paddy production level from farm, environmental and cultivation characteristics.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    a, b, c = st.columns([1.05, 1, 1])
    with a:
        st.markdown(
            f"""
            <div class="prediction-panel">
                <div class="prediction-label">Predicted Yield Level</div>
                <div class="prediction-value">{prediction}</div>
                <div class="pill">{MODEL_SHORT.get(selected_model_name, selected_model_name)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with b:
        st.metric("Prediction Confidence", f"{confidence:.1%}")
    with c:
        yield_range = bundle.get("yield_ranges", {}).get(prediction, "")
        st.metric("Production Band", yield_range if yield_range else prediction)

    st.markdown('<div class="section-title">Class Probability</div>', unsafe_allow_html=True)
    prob_df = pd.DataFrame({
        "Yield Level": class_labels,
        "Probability": [prob_map.get(c, 0.0) for c in class_labels],
    })
    fig = px.bar(
        prob_df,
        x="Probability",
        y="Yield Level",
        orientation="h",
        text=prob_df["Probability"].map(lambda x: f"{x:.1%}"),
        color="Yield Level",
        color_discrete_map={
            "Low": "#5B70F5",
            "Moderate": "#16C784",
            "High": "#A35FEF",
            "Very High": "#FFB800",
        },
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_xaxes(tickformat=".0%", range=[0, max(1.0, prob_df["Probability"].max() * 1.15)])
    fig.update_layout(yaxis_title="", xaxis_title="Probability", showlegend=False)
    clean_plot_layout(fig, height=360, margin=dict(l=20, r=70, t=20, b=45), legend=False)
    st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)

# ============================================================
# 2. Data Understanding — screenshot-like Preprocessing Evidence tabs
# ============================================================
with main_tabs[1]:
    st.markdown('<div class="section-title">Dataset Overview</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Records", f"{len(raw_df):,}")
    m2.metric("Variables", f"{raw_df.shape[1]-1}")  # excludes generated category
    m3.metric("Missing Cells", f"{int(raw_df.drop(columns=['Paddy Yield Category']).isna().sum().sum()):,}")
    m4.metric("Duplicate Rows", f"{int(raw_df.drop(columns=['Paddy Yield Category']).duplicated().sum()):,}")

    with st.expander("View dataset preview", expanded=False):
        st.dataframe(raw_df.drop(columns=["Paddy Yield Category"]).head(10), use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">Preprocessing Evidence</div>', unsafe_allow_html=True)
    du_tabs = st.tabs([
        "Class Distribution",
        "Correlation",
        "Feature Distributions",
        "Categorical Yield Rates",
        "Scaling",
        "Feature Selection",
    ])

    # Class distribution — reference-style bar + notebook pie chart
    with du_tabs[0]:
        order = ["Low", "Moderate", "High", "Very High"]
        counts = raw_df["Paddy Yield Category"].value_counts().reindex(order).fillna(0).astype(int)
        dist = pd.DataFrame({"Yield Level": order, "Count": counts.values})
        dist["Percent"] = dist["Count"] / dist["Count"].sum()

        fig = px.bar(
            dist,
            x="Yield Level",
            y="Count",
            text=[f"{n:,}<br>({p:.1%})" for n, p in zip(dist["Count"], dist["Percent"])],
            color="Yield Level",
            color_discrete_map={
                "Low": "#2ECC71",
                "Moderate": "#5B70F5",
                "High": "#A35FEF",
                "Very High": "#EF4B3E",
            },
        )
        fig.update_traces(textposition="outside", textfont_size=15)
        fig.update_layout(
            title=dict(text="Paddy Yield Class Distribution", x=0.5, font=dict(size=28)),
            xaxis_title="Paddy Yield Category",
            yaxis_title="Count",
            showlegend=False,
        )
        clean_plot_layout(fig, height=540, margin=dict(l=55, r=25, t=95, b=65), legend=False)
        st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)

        st.markdown('<div class="section-title">Paddy Yield Category Share</div>', unsafe_allow_html=True)
        pie_col, table_col = st.columns([1.35, 0.85], gap="large")
        with pie_col:
            pie_fig = go.Figure(
                data=[go.Pie(
                    labels=[
                        "Low: 0–10,000",
                        "Moderate: 10,001–20,000",
                        "High: 20,001–30,000",
                        "Very High: 30,001–40,000",
                    ],
                    values=dist["Count"],
                    textinfo="percent",
                    hovertemplate="%{label}<br>Count: %{value:,}<br>Share: %{percent}<extra></extra>",
                    pull=[0.035, 0.035, 0.035, 0.035],
                    sort=False,
                    marker=dict(
                        colors=["#2ECC71", "#5B70F5", "#A35FEF", "#EF4B3E"],
                        line=dict(color="#FFFFFF", width=2),
                    ),
                )]
            )
            pie_fig.update_layout(
                title=dict(text=f"Share of plots in each paddy yield category (n={len(raw_df):,})", x=0.5, font=dict(size=18)),
                legend=dict(orientation="h", yanchor="top", y=-0.08, xanchor="center", x=0.5),
                margin=dict(l=35, r=35, t=75, b=105),
                height=510,
                paper_bgcolor="white",
                font=dict(family="Arial, sans-serif", size=12, color="#343a46"),
            )
            st.plotly_chart(pie_fig, use_container_width=True, config=PLOT_CONFIG)
        with table_col:
            share_table = dist.copy()
            share_table["Range (Kg)"] = ["0–10,000", "10,001–20,000", "20,001–30,000", "30,001–40,000"]
            share_table["Share"] = share_table["Percent"].map(lambda v: f"{v:.2%}")
            st.markdown("#### Class Summary")
            st.dataframe(
                share_table[["Yield Level", "Range (Kg)", "Count", "Share"]],
                use_container_width=True,
                hide_index=True,
                height=310,
            )
            st.caption("This is the same four-class target definition used in the supplied notebook.")

    # Correlation map — screenshot-style readable map + full notebook 33-input map
    with du_tabs[1]:
        st.markdown("#### Correlation Map")

        # Build the notebook-style 33-input table: 25 numeric + 8 categorical, one column per feature.
        heat_features = [c for c in selected_features if c in raw_df.columns]
        heat_df = raw_df[heat_features + ["Paddy yield(in Kg)"]].copy()
        for col in heat_features:
            if not pd.api.types.is_numeric_dtype(heat_df[col]):
                heat_df[col] = pd.factorize(heat_df[col])[0]
        full_corr = heat_df.corr(numeric_only=True)

        # First plot: a compact 10-variable heatmap, matching the reference dashboard's visual scale.
        target_name = "Paddy yield(in Kg)"
        target_rank = full_corr[target_name].drop(target_name).abs().sort_values(ascending=False)
        compact_features = target_rank.head(9).index.tolist() + [target_name]
        compact_corr = full_corr.loc[compact_features, compact_features]

        fig = px.imshow(
            compact_corr,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            color_continuous_midpoint=0,
        )
        fig.update_layout(
            title=dict(text="Correlation Map — Key Variables", x=0.5, font=dict(size=22)),
            coloraxis_colorbar=dict(title="", thickness=24, len=0.88),
        )
        fig.update_xaxes(tickangle=-90, title="")
        fig.update_yaxes(title="")
        clean_plot_layout(fig, height=720, margin=dict(l=195, r=55, t=75, b=210), legend=False)
        st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)

        st.markdown('<div class="section-title">Full Correlation Heatmap — 33 Input Variables vs Paddy Yield</div>', unsafe_allow_html=True)
        full_fig = px.imshow(
            full_corr,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="Greens",
            zmin=-1,
            zmax=1,
        )
        full_fig.update_layout(
            title=dict(text="Correlation heatmap — 33 input variables vs. Paddy yield(in Kg)", x=0.5, font=dict(size=19)),
            coloraxis_colorbar=dict(title="Correlation", thickness=21, len=0.90),
        )
        full_fig.update_traces(textfont=dict(size=7))
        full_fig.update_xaxes(tickangle=-90, title="", tickfont=dict(size=8))
        full_fig.update_yaxes(title="", tickfont=dict(size=8))
        clean_plot_layout(full_fig, height=1060, margin=dict(l=220, r=50, t=80, b=235), legend=False)
        st.plotly_chart(full_fig, use_container_width=True, config=PLOT_CONFIG)
        st.caption("Categorical inputs are factorised to keep one column per feature, matching the correlation-heatmap method in the supplied notebook.")

    # Feature distributions
    with du_tabs[2]:
        candidates = [
            "Trash(in bundles)",
            "Relative Humidity_D91_D120",
            "Max temp_D61_D90",
            "Inst Wind Speed_D31_D60(in Knots)",
            "Min temp_D61_D90",
            "30_50DRain( in mm)",
        ]
        candidates = [c for c in candidates if c in raw_df.columns]
        rows = [candidates[i:i+3] for i in range(0, len(candidates), 3)]
        for row in rows:
            cols = st.columns(len(row))
            for col_ui, feature in zip(cols, row):
                with col_ui:
                    fig = px.histogram(raw_df, x=feature, nbins=25)
                    fig.update_layout(title=dict(text=feature, x=0.5, font=dict(size=14)), xaxis_title="", yaxis_title="Count", showlegend=False)
                    clean_plot_layout(fig, height=300, margin=dict(l=35, r=15, t=55, b=40), legend=False)
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Categorical yield rates — 2x2, screenshot-like
    with du_tabs[3]:
        cat_show = [c for c in ["Agriblock", "Variety", "Soil Types", "Nursery"] if c in raw_df.columns]
        for i in range(0, len(cat_show), 2):
            left, right = st.columns(2)
            for box, feature in zip([left, right], cat_show[i:i+2]):
                with box:
                    g = (
                        raw_df.groupby([feature, "Paddy Yield Category"], observed=False)
                        .size()
                        .rename("Count")
                        .reset_index()
                    )
                    totals = g.groupby(feature)["Count"].transform("sum")
                    g["Percentage"] = np.where(totals > 0, g["Count"] / totals * 100, 0)
                    fig = px.bar(
                        g,
                        x=feature,
                        y="Percentage",
                        color="Paddy Yield Category",
                        category_orders={"Paddy Yield Category": ["Low", "Moderate", "High", "Very High"]},
                        color_discrete_map={
                            "Low": "#2ECC71",
                            "Moderate": "#5B70F5",
                            "High": "#A35FEF",
                            "Very High": "#EF4B3E",
                        },
                        title=f"Paddy Yield Level by {feature}",
                    )
                    fig.update_layout(yaxis_title="Percentage (%)", legend_title_text="Yield Level")
                    clean_plot_layout(fig, height=370, margin=dict(l=45, r=15, t=60, b=80), legend=True)
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Scaling — 3 before + 3 after, same visual pattern as screenshot
    with du_tabs[4]:
        scale_features = [c for c in [
            "Trash(in bundles)",
            "Relative Humidity_D91_D120",
            "Inst Wind Speed_D31_D60(in Knots)",
        ] if c in df.columns]
        X_scale = df[scale_features].copy()
        y_scale = df["Paddy Yield Category"].astype(str)
        X_train_s, _, _, _ = train_test_split(
            X_scale, y_scale, test_size=0.20, random_state=42, stratify=y_scale
        )
        scaler = StandardScaler().fit(X_train_s)
        scaled = pd.DataFrame(scaler.transform(X_train_s), columns=scale_features)

        st.markdown("#### Before vs After StandardScaler (Training Data Only)")
        top_cols = st.columns(3)
        for box, feature in zip(top_cols, scale_features):
            with box:
                fig = px.histogram(X_train_s, x=feature, nbins=25)
                fig.update_layout(title=dict(text=f"{feature}<br>(Before)", x=0.5, font=dict(size=12)), xaxis_title="", yaxis_title="")
                clean_plot_layout(fig, height=280, margin=dict(l=25, r=10, t=55, b=35), legend=False)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        bottom_cols = st.columns(3)
        for box, feature in zip(bottom_cols, scale_features):
            with box:
                fig = px.histogram(scaled, x=feature, nbins=25)
                fig.update_layout(title=dict(text=f"{feature}<br>(After)", x=0.5, font=dict(size=12)), xaxis_title="", yaxis_title="")
                clean_plot_layout(fig, height=280, margin=dict(l=25, r=10, t=55, b=35), legend=False)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Feature selection evidence — screenshot layout, but uses actual redundancy analysis instead of claiming RFECV
    with du_tabs[5]:
        st.markdown("#### Correlation and Redundancy-Based Feature Selection")
        c_left, c_right = st.columns([1.05, 1.2])
        with c_left:
            tmp = raw_df.select_dtypes(include=np.number).corr(numeric_only=True)["Paddy yield(in Kg)"].drop("Paddy yield(in Kg)")
            top15 = tmp.abs().sort_values(ascending=False).head(15).index
            plot_df = pd.DataFrame({
                "Feature": top15,
                "Correlation": tmp.loc[top15].values,
            }).sort_values("Correlation")
            fig = px.bar(plot_df, x="Correlation", y="Feature", orientation="h", text=plot_df["Correlation"].map(lambda x: f"{x:.3f}"))
            fig.update_traces(textposition="outside")
            fig.update_layout(title=dict(text="Top 15 Correlations with Paddy Yield", x=0.5, font=dict(size=16)), xaxis_title="Pearson Correlation", yaxis_title="", showlegend=False)
            clean_plot_layout(fig, height=510, margin=dict(l=175, r=45, t=65, b=45), legend=False)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        with c_right:
            excluded = bundle.get("excluded_features", [])
            retained = selected_features
            selection_df = pd.DataFrame({
                "Feature": retained + excluded,
                "Status": ["Retained"] * len(retained) + ["Removed (Redundant)"] * len(excluded),
                "Ranking": [1] * len(retained) + [2] * len(excluded),
            })
            # display 28 most informative rows for readability
            selection_df = selection_df.head(28)
            fig = px.bar(
                selection_df,
                x="Ranking",
                y="Feature",
                orientation="h",
                color="Status",
                color_discrete_map={"Retained": "#28B463", "Removed (Redundant)": "#EF4B3E"},
                category_orders={"Status": ["Retained", "Removed (Redundant)"]},
            )
            fig.update_layout(title=dict(text="Feature Selection Status", x=0.5, font=dict(size=16)), xaxis=dict(range=[0, 2.2], tickvals=[1,2], ticktext=["Retained", "Removed"]), yaxis_title="", legend_title_text="")
            clean_plot_layout(fig, height=510, margin=dict(l=190, r=20, t=65, b=60), legend=True)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.caption("The notebook identifies a group of variables that are exact proportional transformations of Hectares; those redundant variables are excluded, while the remaining predictors are retained.")

# ============================================================
# 3. Model Performance — close to screenshot scorecard + CV + training tabs
# ============================================================
with main_tabs[2]:
    st.markdown(
        """
        <div class="hero">
            <h1>Model Performance</h1>
            <p>Evaluation of the four classification models using held-out test metrics, 5-fold cross-validation and training visualisations.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Model Performance Comparison Table</div>', unsafe_allow_html=True)
    st.caption("Comparison values are taken directly from the Evaluation section of the supplied ABCCC (2).ipynb.")

    compare_style = (
        NOTEBOOK_COMPARISON.style
        .format({
            "Accuracy": "{:.2f}%",
            "Macro Precision": "{:.2f}%",
            "Macro Recall": "{:.2f}%",
            "Macro F1-score": "{:.2f}%",
        })
        .highlight_max(
            subset=["Accuracy", "Macro Precision", "Macro Recall", "Macro F1-score"],
            color="#d9f0df",
        )
    )
    st.dataframe(compare_style, use_container_width=True, hide_index=True, height=215)

    st.markdown('<div class="section-title">Performance Comparison</div>', unsafe_allow_html=True)
    perf_long = NOTEBOOK_COMPARISON.melt(
        id_vars="Model",
        value_vars=["Accuracy", "Macro Precision", "Macro Recall", "Macro F1-score"],
        var_name="Metric",
        value_name="Score (%)",
    )
    fig = px.bar(
        perf_long,
        x="Model",
        y="Score (%)",
        color="Metric",
        barmode="group",
        text=perf_long["Score (%)"].map(lambda v: f"{v:.2f}%"),
        color_discrete_sequence=["#5B70F5", "#F05A3D", "#16C784", "#A35FEF"],
    )
    fig.update_traces(textposition="outside", textfont_size=9)
    fig.update_yaxes(range=[85, 100], dtick=2)
    fig.update_layout(
        title=dict(text="Model Performance Comparison", x=0.5, font=dict(size=19)),
        xaxis_title="Model",
        yaxis_title="Score (%)",
        legend_title_text="",
    )
    clean_plot_layout(fig, height=500, margin=dict(l=48, r=20, t=70, b=100), legend=True)
    st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)

    # Keep the model artifact diagnostics available below the notebook comparison.
    with st.expander("Detailed held-out scorecard (includes AUC and Log Loss)", expanded=False):
        score_cols = ["Model", "Accuracy", "Precision", "Recall", "F1 Score", "AUC", "Log Loss"]
        score = metrics_df[score_cols].copy()
        score["Model"] = score["Model"].map(lambda x: MODEL_SHORT.get(x, x))
        detailed_style = (
            score.style
            .format({c: "{:.4f}" for c in score_cols if c != "Model"})
            .highlight_max(subset=["Accuracy", "Precision", "Recall", "F1 Score", "AUC"], color="#d9f0df")
            .highlight_min(subset=["Log Loss"], color="#d9f0df")
        )
        st.dataframe(detailed_style, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">K-Fold Cross-Validation</div>', unsafe_allow_html=True)
    st.write("5-Fold Stratified Cross-Validation. Each model is evaluated across the same five folds on the prepared training data.")
    cv_rows = []
    for model_name in MODEL_ORDER:
        vals = cv_folds.get(model_name, [])
        if not vals:
            continue
        row = {"Model": MODEL_SHORT.get(model_name, model_name)}
        for i, v in enumerate(vals, 1):
            row[f"Fold {i}"] = float(v)
        row["Mean"] = float(np.mean(vals))
        row["Std"] = float(np.std(vals))
        cv_rows.append(row)
    cv_df = pd.DataFrame(cv_rows)
    cv_styled = (
        cv_df.style
        .format({c: "{:.4f}" for c in cv_df.columns if c != "Model"})
        .highlight_max(subset=["Mean"], color="#d9f0df")
        .highlight_min(subset=["Std"], color="#d9f0df")
    )
    st.dataframe(cv_styled, use_container_width=True, hide_index=True)

    cv_long_rows = []
    for model_name in MODEL_ORDER:
        for i, value in enumerate(cv_folds.get(model_name, []), 1):
            cv_long_rows.append({"Model": MODEL_SHORT.get(model_name, model_name), "Fold": f"Fold {i}", "F1 Score": float(value)})
    cv_long = pd.DataFrame(cv_long_rows)
    fig = px.bar(
        cv_long,
        x="Fold",
        y="F1 Score",
        color="Model",
        barmode="group",
        text_auto=".3f",
        color_discrete_sequence=["#5B70F5", "#F05A3D", "#16C784", "#A35FEF"],
    )
    fig.update_traces(textposition="outside", textfont_size=9)
    fig.update_yaxes(range=[0, 1.08])
    fig.update_layout(title=dict(text="F1-Score per Fold (5-Fold Stratified CV)", x=0.0, font=dict(size=16)), legend_title_text="")
    clean_plot_layout(fig, height=450, margin=dict(l=40, r=20, t=60, b=95), legend=True)
    st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)

    best_cv_row = cv_df.loc[cv_df["Mean"].idxmax()]
    st.info(f"Best K-Fold CV Model: {best_cv_row['Model']} (Mean F1 = {best_cv_row['Mean']:.4f} ± {best_cv_row['Std']:.4f})")

    st.markdown('<div class="section-title">Training Visualizations</div>', unsafe_allow_html=True)
    viz_tabs = st.tabs([
        "Confusion Matrices",
        "ROC Curves",
        "PR Curves",
        "Feature Importance",
        "Learning Curves",
        "Tuning Evidence",
    ])

    # Confusion matrices: 3 on top + 1 underneath, like screenshot
    with viz_tabs[0]:
        def confusion_fig(model_name):
            cm = np.asarray(bundle["confusion_matrices"][model_name])
            mrow = metrics_df[metrics_df["Model"] == model_name].iloc[0]
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=class_labels,
                y=class_labels,
                colorscale="Blues",
                showscale=False,
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 14},
                hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
            ))
            fig.update_layout(
                title=dict(
                    text=f"{MODEL_SHORT.get(model_name, model_name)}<br><sup>Acc={mrow['Accuracy']:.3f}  F1={mrow['F1 Score']:.3f}</sup>",
                    x=0.5,
                    font=dict(size=14),
                ),
                xaxis_title="Predicted",
                yaxis_title="Actual",
                height=310,
                margin=dict(l=45, r=10, t=72, b=45),
                paper_bgcolor="white",
                plot_bgcolor="white",
                font=dict(family="Arial", size=11),
            )
            fig.update_yaxes(autorange="reversed")
            return fig

        row1 = st.columns(3)
        for box, model_name in zip(row1, MODEL_ORDER[:3]):
            with box:
                st.plotly_chart(confusion_fig(model_name), use_container_width=True, config={"displayModeBar": False})
        if len(MODEL_ORDER) > 3:
            row2 = st.columns(3)
            with row2[0]:
                st.plotly_chart(confusion_fig(MODEL_ORDER[3]), use_container_width=True, config={"displayModeBar": False})

    # ROC curves
    with viz_tabs[1]:
        roc_model = st.selectbox("ROC model", MODEL_ORDER, format_func=lambda x: MODEL_SHORT.get(x, x), key="roc_model")
        roc_obj = bundle["roc_data"][roc_model]
        fig = go.Figure()
        # support several possible saved structures
        if isinstance(roc_obj, dict):
            for label, item in roc_obj.items():
                if isinstance(item, dict) and "fpr" in item and "tpr" in item:
                    auc_v = item.get("auc", np.nan)
                    fig.add_trace(go.Scatter(x=item["fpr"], y=item["tpr"], mode="lines", name=f"{label} (AUC={auc_v:.3f})" if pd.notna(auc_v) else str(label)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(dash="dash")))
        fig.update_layout(title=f"Multiclass ROC Curves — {MODEL_SHORT.get(roc_model, roc_model)}", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        fig.update_xaxes(range=[0,1]); fig.update_yaxes(range=[0,1.02])
        clean_plot_layout(fig, height=520, margin=dict(l=50, r=20, t=65, b=85), legend=True)
        st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)

    # PR curves
    with viz_tabs[2]:
        pr_model = st.selectbox("PR model", MODEL_ORDER, format_func=lambda x: MODEL_SHORT.get(x, x), key="pr_model")
        pr_obj = bundle["pr_data"][pr_model]
        fig = go.Figure()
        if isinstance(pr_obj, dict):
            for label, item in pr_obj.items():
                if isinstance(item, dict) and "precision" in item and "recall" in item:
                    fig.add_trace(go.Scatter(x=item["recall"], y=item["precision"], mode="lines", name=str(label)))
        fig.update_layout(title=f"Precision-Recall Curves — {MODEL_SHORT.get(pr_model, pr_model)}", xaxis_title="Recall", yaxis_title="Precision")
        fig.update_xaxes(range=[0,1]); fig.update_yaxes(range=[0,1.02])
        clean_plot_layout(fig, height=520, margin=dict(l=50, r=20, t=65, b=85), legend=True)
        st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)

    # Feature importance
    with viz_tabs[3]:
        fi = pd.DataFrame(bundle["rf_feature_importance"]).copy()
        # normalize common column names
        if "Feature" not in fi.columns:
            fi.columns = ["Feature", "Importance"] + list(fi.columns[2:])
        importance_col = next((c for c in fi.columns if "Importance" in str(c) and c != "Feature"), fi.columns[1])
        fi = fi.sort_values(importance_col, ascending=False).head(15).sort_values(importance_col)
        fig = px.bar(fi, x=importance_col, y="Feature", orientation="h", text=fi[importance_col].map(lambda x: f"{float(x):.4f}"))
        fig.update_traces(textposition="outside")
        fig.update_layout(title="Random Forest — Top 15 Feature Importances", xaxis_title="Importance", yaxis_title="", showlegend=False)
        clean_plot_layout(fig, height=560, margin=dict(l=190, r=55, t=65, b=50), legend=False)
        st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)

    # Learning curves
    with viz_tabs[4]:
        lc = bundle["rf_learning_curve"]
        train_sizes = np.asarray(lc["train_sizes"])
        train_mean = np.asarray(lc["train_mean"])
        val_mean = np.asarray(lc["val_mean"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train_sizes, y=train_mean, mode="lines+markers", name="Training F1"))
        fig.add_trace(go.Scatter(x=train_sizes, y=val_mean, mode="lines+markers", name="Validation F1"))
        fig.update_layout(title="Random Forest Learning Curve", xaxis_title="Training Samples", yaxis_title="Macro F1")
        fig.update_yaxes(range=[0,1.05])
        clean_plot_layout(fig, height=500, margin=dict(l=55, r=20, t=65, b=80), legend=True)
        st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)

    # Tuning evidence
    with viz_tabs[5]:
        tune_df = pd.DataFrame(bundle["rf_tuning_evidence"])
        st.markdown("#### Random Forest Hyperparameter Tuning Evidence")
        st.dataframe(tune_df, use_container_width=True, hide_index=True)
        if bundle.get("best_params"):
            st.markdown("**Best Random Forest parameters**")
            st.json(bundle["best_params"], expanded=False)

# ============================================================
# 4. Data Preparation
# ============================================================
with main_tabs[3]:
    st.markdown(
        """
        <div class="hero">
            <h1>Data Preparation</h1>
            <p>Cleaning, target construction, feature selection, train-test split and preprocessing used before modelling.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    prep_rows = [
        ("1", "Clean column and categorical values", "Standardised column names and categorical labels."),
        ("2", "Check missing values", "No missing cells were found in the original dataset."),
        ("3", "Remove duplicate records", f"{raw_df.drop(columns=['Paddy Yield Category']).duplicated().sum():,} duplicate rows removed; {len(df):,} rows retained."),
        ("4", "Remove unsuitable/redundant variables", f"{len(bundle.get('excluded_features', []))} highly redundant variables excluded after correlation/redundancy analysis."),
        ("5", "Create multiclass target", "Low: 0–10,000 kg; Moderate: 10,001–20,000 kg; High: 20,001–30,000 kg; Very High: 30,001–40,000 kg."),
        ("6", "Separate X and y", f"{len(selected_features)} predictor variables retained for modelling."),
        ("7", "Train-test split", "Stratified train-test split with random_state=42."),
        ("8", "Encode and scale", "Categorical variables are one-hot encoded; numerical variables are standardised inside the modelling pipeline where required."),
        ("9", "Prevent data leakage", "Paddy yield is used only to create the target and is not included as an input predictor."),
    ]
    prep_df = pd.DataFrame(prep_rows, columns=["Step", "Preparation Stage", "Evidence"])
    st.dataframe(prep_df, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">Final Prepared Feature Set</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Numerical predictors**")
        st.dataframe(pd.DataFrame({"Feature": num_cols}), use_container_width=True, hide_index=True, height=470)
    with c2:
        st.write("**Categorical predictors**")
        st.dataframe(pd.DataFrame({"Feature": cat_cols}), use_container_width=True, hide_index=True, height=470)

# ============================================================
# 5. About
# ============================================================
with main_tabs[4]:
    st.markdown(
        """
        <div class="hero">
            <h1>About</h1>
            <p>Paddy yield classification dashboard for BMDS2003 Data Science.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### Objective")
    st.write("Given the characteristics and conditions of a farm, predict the expected level of total paddy production.")
    st.markdown("### Target")
    st.write("Paddy yield is converted into four classes: Low, Moderate, High and Very High.")
    st.markdown("### Models")
    st.write("Logistic Regression, K-Nearest Neighbours, Random Forest and Artificial Neural Network.")
