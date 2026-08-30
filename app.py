from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "paddydataset.csv"
BUNDLE_PATH = BASE_DIR / "paddy_dashboard_bundle.joblib"

st.set_page_config(
    page_title="Paddy Yield Prediction",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --accent: #ef4444;
        --ink: #172033;
        --muted: #718096;
        --line: #e6eaf0;
        --panel: #f7f9fb;
    }
    .stApp { background: #ffffff; }
    .block-container { max-width: 1280px; padding-top: 1.45rem; padding-bottom: 3rem; }
    section[data-testid="stSidebar"] {
        background: #f3f5f7;
        border-right: 1px solid #e2e7ec;
    }
    section[data-testid="stSidebar"] > div { padding-top: 1.2rem; }
    h1, h2, h3 { color: var(--ink); letter-spacing: -0.02em; }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 13px;
        padding: 16px 18px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04);
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 12px;
        overflow: hidden;
    }
    button[data-baseweb="tab"] { font-size: 0.88rem; }
    [data-baseweb="tab-highlight"] { background-color: var(--accent) !important; }
    .hero {
        background: linear-gradient(112deg, #102d35 0%, #224956 55%, #315f6b 100%);
        border-radius: 17px;
        padding: 34px 38px;
        margin: 4px 0 22px 0;
        box-shadow: 0 14px 30px rgba(16,45,53,.18);
    }
    .hero h1 { color: white !important; margin: 0; font-size: 2rem; }
    .hero p { color: #dce9ec; margin: 11px 0 0; font-size: 1rem; }
    .side-title { text-align:center; padding: 14px 4px 20px; }
    .side-title h2 { margin-bottom: 5px; font-size: 1.55rem; line-height: 1.15; }
    .side-title p { color:#7b8794; font-size:.78rem; margin:0; }
    .soft-card {
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 18px 20px;
        background: #fff;
    }
    .prediction-card {
        border-radius: 16px;
        padding: 22px;
        background: linear-gradient(135deg, #f8fafc 0%, #eef4f5 100%);
        border: 1px solid #dbe5e8;
        min-height: 155px;
    }
    .badge {
        display:inline-block;
        padding: 7px 14px;
        border-radius: 999px;
        font-weight: 800;
        font-size: .95rem;
        margin-top: 7px;
    }
    .tiny-note { color:#718096; font-size:.82rem; }
    .section-rule { border-top: 1px solid #e6eaf0; margin: 18px 0 14px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_bundle():
    return joblib.load(BUNDLE_PATH)


@st.cache_data
def load_data():
    raw_data = pd.read_csv(DATA_PATH)
    raw_data.columns = raw_data.columns.str.strip()

    # Data Understanding uses the original dataset, matching the notebook.
    raw_data["Paddy Yield Category"] = pd.cut(
        raw_data["Paddy yield(in Kg)"],
        bins=[0, 10000, 20000, 30000, 40000],
        labels=["Low", "Moderate", "High", "Very High"],
        right=False,
    )

    # Data Preparation and modelling use the cleaned dataset after duplicates are removed.
    cleaned_data = raw_data.drop_duplicates(
        subset=[c for c in raw_data.columns if c != "Paddy Yield Category"]
    ).reset_index(drop=True)

    return raw_data, cleaned_data


try:
    bundle = load_bundle()
    raw_df, df = load_data()
except Exception as exc:
    st.error(
        "The dashboard files could not be loaded. Keep app.py, paddydataset.csv, "
        "and paddy_dashboard_bundle.joblib in the same GitHub folder."
    )
    st.exception(exc)
    st.stop()

models = bundle["models"]
class_labels = bundle["class_labels"]
yield_ranges = bundle["yield_ranges"]
num_cols = bundle["numeric_cols"]
cat_cols = bundle["categorical_cols"]

CLASS_COLORS = {
    "Low": "#3b82f6",
    "Moderate": "#22c55e",
    "High": "#f59e0b",
    "Very High": "#ef4444",
}
BADGE_STYLE = {
    "Low": ("#e8f1ff", "#1d4ed8"),
    "Moderate": ("#e8f8ef", "#15803d"),
    "High": ("#fff4d9", "#b45309"),
    "Very High": ("#ffe7e7", "#b91c1c"),
}

# --------------------------- Sidebar ---------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="side-title">
            <h2>Paddy Yield<br>Prediction</h2>
            <p>BMDS2003 Data Science</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    selected_model_name = st.selectbox(
        "Prediction model",
        list(models.keys()),
        index=list(models.keys()).index("Random Forest (Tuned)"),
    )

    st.markdown("### Farm Information")
    user_values = {}
    for col in ["Agriblock", "Variety", "Soil Types", "Nursery"]:
        user_values[col] = st.selectbox(col, bundle["categorical_options"][col])

    def numeric_widget(col, key_prefix="input"):
        stats = bundle["numeric_stats"][col]
        span = max(stats["max"] - stats["min"], 1.0)
        step = max(span / 100.0, 0.01)
        return st.number_input(
            col,
            min_value=float(stats["min"]),
            max_value=float(stats["max"]),
            value=float(stats["median"]),
            step=float(step),
            key=f"{key_prefix}_{col}",
        )

    rain_cols = [
        "30DRain( in mm)", "30DAI(in mm)", "30_50DRain( in mm)", "30_50DAI(in mm)",
        "51_70DRain(in mm)", "51_70AI(in mm)", "71_105DRain(in mm)", "71_105DAI(in mm)",
    ]
    temp_cols = [
        "Min temp_D1_D30", "Max temp_D1_D30", "Min temp_D31_D60", "Max temp_D31_D60",
        "Min temp_D61_D90", "Max temp_D61_D90", "Min temp_D91_D120", "Max temp_D91_D120",
    ]
    wind_speed_cols = [
        "Inst Wind Speed_D1_D30(in Knots)", "Inst Wind Speed_D31_D60(in Knots)",
        "Inst Wind Speed_D61_D90(in Knots)", "Inst Wind Speed_D91_D120(in Knots)",
    ]
    wind_dir_cols = [
        "Wind Direction_D1_D30", "Wind Direction_D31_D60",
        "Wind Direction_D61_D90", "Wind Direction_D91_D120",
    ]
    humidity_cols = [
        "Relative Humidity_D1_D30", "Relative Humidity_D31_D60",
        "Relative Humidity_D61_D90", "Relative Humidity_D91_D120",
    ]

    with st.expander("Rainfall & moisture", expanded=False):
        for col in rain_cols:
            user_values[col] = numeric_widget(col)

    with st.expander("Temperature", expanded=False):
        for col in temp_cols:
            user_values[col] = numeric_widget(col)

    with st.expander("Wind conditions", expanded=False):
        for col in wind_speed_cols:
            user_values[col] = numeric_widget(col)
        for col in wind_dir_cols:
            user_values[col] = st.selectbox(col, bundle["categorical_options"][col])

    with st.expander("Relative humidity", expanded=False):
        for col in humidity_cols:
            user_values[col] = numeric_widget(col)

    with st.expander("Production indicator", expanded=True):
        user_values["Trash(in bundles)"] = numeric_widget("Trash(in bundles)")

    st.caption("Prediction updates automatically when an input changes.")

# Make sure input order exactly matches training features.
input_df = pd.DataFrame([{col: user_values[col] for col in bundle["selected_features"]}])
selected_model = models[selected_model_name]
prediction = str(selected_model.predict(input_df)[0])
probabilities = selected_model.predict_proba(input_df)[0]
model_class_order = list(selected_model.named_steps["model"].classes_)
prob_map = {str(c): float(p) for c, p in zip(model_class_order, probabilities)}
confidence = prob_map[prediction]

# --------------------------- Main Navigation ---------------------------
main_tabs = st.tabs(["Prediction", "Data Understanding", "Data Preparation", "Model Performance", "About"])

with main_tabs[0]:
    st.markdown(
        """
        <div class="hero">
            <h1>🌾 Paddy Yield Prediction</h1>
            <p>Predict Low, Moderate, High or Very High total paddy production from the farm characteristics retained in your notebook.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    bg, fg = BADGE_STYLE[prediction]
    c1, c2, c3 = st.columns([1.15, 1, 1])
    with c1:
        st.markdown(
            f"""
            <div class="prediction-card">
                <div class="tiny-note">Predicted Production Level</div>
                <div class="badge" style="background:{bg};color:{fg};">{prediction}</div>
                <div style="font-size:1.15rem;font-weight:700;margin-top:15px;color:#172033;">{yield_ranges[prediction]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.metric("Prediction Confidence", f"{confidence:.1%}")
        st.metric("Selected Model", selected_model_name.replace(" (Tuned)", ""))
    with c3:
        perf = pd.DataFrame(bundle["metrics"])
        row = perf.loc[perf["Model"] == selected_model_name].iloc[0]
        st.metric("Test Accuracy", f"{row['Accuracy']:.2%}")
        st.metric("Macro F1", f"{row['F1 Score']:.4f}")

    st.markdown("### Class Probability")
    prob_df = pd.DataFrame({
        "Paddy Yield Category": class_labels,
        "Probability": [prob_map.get(c, 0.0) for c in class_labels],
    })
    fig = px.bar(
        prob_df,
        x="Probability",
        y="Paddy Yield Category",
        orientation="h",
        text=prob_df["Probability"].map(lambda x: f"{x:.1%}"),
        color="Paddy Yield Category",
        color_discrete_map=CLASS_COLORS,
        category_orders={"Paddy Yield Category": class_labels},
    )
    fig.update_layout(
        showlegend=False,
        height=330,
        margin=dict(l=10, r=30, t=10, b=20),
        xaxis_tickformat=".0%",
        xaxis_range=[0, 1],
        yaxis_title="",
        xaxis_title="Predicted Probability",
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Current Input Snapshot")
    snap_cols = ["Agriblock", "Variety", "Soil Types", "Nursery", "Trash(in bundles)",
                 "Min temp_D1_D30", "Relative Humidity_D91_D120"]
    snapshot = input_df[snap_cols].T.reset_index()
    snapshot.columns = ["Feature", "Current Value"]
    st.dataframe(snapshot, use_container_width=True, hide_index=True)

with main_tabs[1]:
    st.markdown(
        """
        <div class="hero">
            <h1>Data Understanding</h1>
            <p>Understand the original Paddy dataset before cleaning and modelling: structure, data quality, descriptive statistics, correlation, EDA and target distribution.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    original_feature_cols = [c for c in raw_df.columns if c != "Paddy Yield Category"]
    raw_numeric_cols = raw_df[original_feature_cols].select_dtypes(include="number").columns.tolist()
    raw_categorical_cols = raw_df[original_feature_cols].select_dtypes(include="object").columns.tolist()

    # Notebook evidence: original dataset contains 2,789 rows and 45 variables.
    o1, o2, o3, o4 = st.columns(4)
    o1.metric("Original Records", f"{len(raw_df):,}")
    o2.metric("Variables", f"{len(original_feature_cols)}")
    o3.metric("Numerical Variables", f"{len(raw_numeric_cols)}")
    o4.metric("Categorical Variables", f"{len(raw_categorical_cols)}")

    understanding_tabs = st.tabs([
        "Dataset Overview",
        "Data Quality",
        "Summary Statistics",
        "Correlation Map",
        "Numeric EDA",
        "Categorical EDA",
        "Target Distribution",
    ])

    with understanding_tabs[0]:
        st.subheader("Dataset Preview")
        st.dataframe(
            raw_df[original_feature_cols].head(10),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Dataset Structure")
        structure_df = pd.DataFrame({
            "Feature": original_feature_cols,
            "Data Type": raw_df[original_feature_cols].dtypes.astype(str).values,
            "Missing Values": raw_df[original_feature_cols].isna().sum().values,
            "Unique Values": raw_df[original_feature_cols].nunique(dropna=True).values,
        })
        st.dataframe(structure_df, use_container_width=True, hide_index=True, height=520)

    with understanding_tabs[1]:
        missing_total = int(raw_df[original_feature_cols].isna().sum().sum())
        duplicate_rows = int(raw_df[original_feature_cols].duplicated().sum())
        duplicate_involved = int(raw_df[original_feature_cols].duplicated(keep=False).sum())
        duplicate_pct = duplicate_rows / len(raw_df) * 100 if len(raw_df) else 0

        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Missing Cells", f"{missing_total:,}")
        q2.metric("Duplicate Rows", f"{duplicate_rows:,}", f"{duplicate_pct:.2f}%")
        q3.metric("Rows Involved in Duplicates", f"{duplicate_involved:,}")
        q4.metric("Rows After Cleaning", f"{len(df):,}")

        st.subheader("Missing Values by Feature")
        missing_df = pd.DataFrame({
            "Feature": original_feature_cols,
            "Missing Values": raw_df[original_feature_cols].isna().sum().values,
        })
        missing_df["Missing Percentage (%)"] = (
            missing_df["Missing Values"] / len(raw_df) * 100
        ).round(2)
        st.dataframe(missing_df, use_container_width=True, hide_index=True, height=450)

        st.subheader("Duplicate Check")
        duplicate_summary = pd.DataFrame({
            "Data Quality Check": [
                "Rows before removing duplicates",
                "Duplicate rows identified",
                "Total rows involved in duplicate groups",
                "Rows after removing duplicates",
                "Remaining duplicate rows",
            ],
            "Count": [
                len(raw_df),
                duplicate_rows,
                duplicate_involved,
                len(df),
                int(df[[c for c in df.columns if c != "Paddy Yield Category"]].duplicated().sum()),
            ],
        })
        st.dataframe(duplicate_summary, use_container_width=True, hide_index=True)

    with understanding_tabs[2]:
        stats_tab1, stats_tab2 = st.tabs(["Numerical Summary", "Categorical Summary"])
        with stats_tab1:
            numerical_summary = raw_df[raw_numeric_cols].describe().T.round(2)
            st.dataframe(numerical_summary, use_container_width=True, height=620)
        with stats_tab2:
            categorical_summary = raw_df[raw_categorical_cols].describe().T
            st.dataframe(categorical_summary, use_container_width=True, height=500)

    with understanding_tabs[3]:
        st.subheader("Correlation Heatmap — 33 Input Variables vs Paddy Yield")
        st.caption(
            "This reproduces the notebook's representative-feature correlation map. "
            "The 8 categorical variables are factorised only for this correlation visualisation."
        )

        corr_features = bundle["selected_features"]
        corr_df = raw_df[corr_features + ["Paddy yield(in Kg)"]].copy()
        for col in bundle["categorical_cols"]:
            corr_df[col] = pd.factorize(corr_df[col])[0]
        full_corr = corr_df.corr(numeric_only=True)

        heatmap = px.imshow(
            full_corr,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="Greens",
            zmin=-1,
            zmax=1,
            title="Correlation Heatmap — 33 Input Variables vs Paddy Yield (Original n=2,789)",
        )
        heatmap.update_traces(textfont_size=8)
        heatmap.update_layout(
            height=1050,
            margin=dict(l=20, r=20, t=70, b=20),
            xaxis_tickangle=-65,
        )
        st.plotly_chart(heatmap, use_container_width=True)

        st.subheader("Correlation of the 25 Numerical Predictors with Paddy Yield")
        numeric_corr = (
            raw_df[bundle["numeric_cols"] + ["Paddy yield(in Kg)"]]
            .corr(numeric_only=True)["Paddy yield(in Kg)"]
            .drop("Paddy yield(in Kg)")
            .sort_values(ascending=False)
            .rename("Correlation")
            .reset_index()
            .rename(columns={"index": "Feature"})
        )
        corr_bar = numeric_corr.sort_values("Correlation")
        fig_corr = px.bar(
            corr_bar,
            x="Correlation",
            y="Feature",
            orientation="h",
            text="Correlation",
            title="Pearson Correlation with Paddy Yield",
        )
        fig_corr.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig_corr.update_layout(height=760, yaxis_title="", xaxis_range=[-1, 1])
        st.plotly_chart(fig_corr, use_container_width=True)

        st.markdown(
            "**Redundancy finding from the notebook:** eleven farm-input variables were exact multiples "
            "of Hectares, so they carried the same underlying plot-size information and were excluded "
            "before modelling. Trash(in bundles) was retained because it was not exactly proportional."
        )

    with understanding_tabs[4]:
        st.subheader("EDA of Numerical Variables")
        family = st.selectbox(
            "Variable group",
            ["Rainfall & AI", "Temperature", "Wind Speed", "Relative Humidity & Trash"],
            key="understanding_numeric_family",
        )
        numeric_groups = {
            "Rainfall & AI": [
                "30DRain( in mm)", "30DAI(in mm)", "30_50DRain( in mm)", "30_50DAI(in mm)",
                "51_70DRain(in mm)", "51_70AI(in mm)", "71_105DRain(in mm)", "71_105DAI(in mm)",
            ],
            "Temperature": [
                "Min temp_D1_D30", "Max temp_D1_D30", "Min temp_D31_D60", "Max temp_D31_D60",
                "Min temp_D61_D90", "Max temp_D61_D90", "Min temp_D91_D120", "Max temp_D91_D120",
            ],
            "Wind Speed": [
                "Inst Wind Speed_D1_D30(in Knots)", "Inst Wind Speed_D31_D60(in Knots)",
                "Inst Wind Speed_D61_D90(in Knots)", "Inst Wind Speed_D91_D120(in Knots)",
            ],
            "Relative Humidity & Trash": [
                "Relative Humidity_D1_D30", "Relative Humidity_D31_D60",
                "Relative Humidity_D61_D90", "Relative Humidity_D91_D120", "Trash(in bundles)",
            ],
        }
        numeric_feature = st.selectbox(
            "Feature",
            numeric_groups[family],
            key="understanding_numeric_feature",
        )
        c1, c2 = st.columns(2)
        with c1:
            hist = px.histogram(
                raw_df,
                x=numeric_feature,
                nbins=30,
                title=f"Distribution of {numeric_feature}",
            )
            hist.update_layout(height=430)
            st.plotly_chart(hist, use_container_width=True)
        with c2:
            scatter = px.scatter(
                raw_df,
                x=numeric_feature,
                y="Paddy yield(in Kg)",
                opacity=0.55,
                title=f"{numeric_feature} vs Paddy Yield",
            )
            scatter.update_layout(height=430)
            st.plotly_chart(scatter, use_container_width=True)

    with understanding_tabs[5]:
        st.subheader("EDA of Categorical Variables")
        categorical_feature = st.selectbox(
            "Categorical feature",
            bundle["categorical_cols"],
            key="understanding_cat_feature",
        )
        c1, c2 = st.columns(2)
        with c1:
            frequency = (
                raw_df[categorical_feature]
                .value_counts(dropna=False)
                .rename_axis(categorical_feature)
                .reset_index(name="Count")
            )
            fig_freq = px.bar(
                frequency,
                x=categorical_feature,
                y="Count",
                text="Count",
                title=f"Frequency Count — {categorical_feature}",
            )
            fig_freq.update_traces(textposition="outside")
            fig_freq.update_layout(height=470)
            st.plotly_chart(fig_freq, use_container_width=True)
        with c2:
            fig_box = px.box(
                raw_df,
                x=categorical_feature,
                y="Paddy yield(in Kg)",
                points=False,
                title=f"Paddy Yield Distribution by {categorical_feature}",
            )
            fig_box.update_layout(height=470)
            st.plotly_chart(fig_box, use_container_width=True)

    with understanding_tabs[6]:
        target_counts = (
            raw_df["Paddy Yield Category"]
            .value_counts()
            .reindex(class_labels)
            .fillna(0)
            .astype(int)
            .reset_index()
        )
        target_counts.columns = ["Paddy Yield Category", "Count"]
        target_counts["Percentage"] = target_counts["Count"] / target_counts["Count"].sum()

        c1, c2 = st.columns(2)
        with c1:
            pie = px.pie(
                target_counts,
                names="Paddy Yield Category",
                values="Count",
                color="Paddy Yield Category",
                color_discrete_map=CLASS_COLORS,
                category_orders={"Paddy Yield Category": class_labels},
                hole=0.35,
                title="Share of Plots in Each Paddy Yield Category",
            )
            pie.update_traces(textposition="inside", textinfo="percent+label")
            pie.update_layout(height=500)
            st.plotly_chart(pie, use_container_width=True)
        with c2:
            bar = px.bar(
                target_counts,
                x="Paddy Yield Category",
                y="Count",
                text=target_counts.apply(
                    lambda r: f"{int(r['Count'])}<br>({r['Percentage']:.1%})", axis=1
                ),
                color="Paddy Yield Category",
                color_discrete_map=CLASS_COLORS,
                category_orders={"Paddy Yield Category": class_labels},
                title="Number of Plots per Yield Category",
            )
            bar.update_traces(textposition="outside")
            bar.update_layout(showlegend=False, height=500)
            st.plotly_chart(bar, use_container_width=True)

        st.caption(
            "Target bins used for classification: Low 0–9,999 kg, Moderate 10,000–19,999 kg, "
            "High 20,000–29,999 kg, and Very High 30,000–39,999 kg."
        )


with main_tabs[2]:
    st.markdown(
        """
        <div class="hero">
            <h1>Data Preparation</h1>
            <p>Cleaning, target creation, feature selection, scaling and preprocessing evidence from the same workflow used in the notebook.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    prep_tabs = st.tabs([
        "Class Distribution",
        "Correlation",
        "Feature Distributions",
        "Categorical Analysis",
        "Scaling",
        "Feature Selection",
    ])

    with prep_tabs[0]:
        counts = (
            df["Paddy Yield Category"].value_counts().reindex(class_labels).fillna(0).astype(int).reset_index()
        )
        counts.columns = ["Paddy Yield Category", "Count"]
        counts["Percent"] = counts["Count"] / counts["Count"].sum()
        fig = px.bar(
            counts,
            x="Paddy Yield Category",
            y="Count",
            text=counts.apply(lambda r: f"{r['Count']}\n({r['Percent']:.1%})", axis=1),
            color="Paddy Yield Category",
            color_discrete_map=CLASS_COLORS,
            category_orders={"Paddy Yield Category": class_labels},
            title="Paddy Yield Category Distribution",
        )
        fig.update_layout(showlegend=False, height=520, xaxis_title="Yield Category")
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("After duplicate removal: 2,338 records. Target bins are 0–9,999, 10,000–19,999, 20,000–29,999 and 30,000–39,999 kg.")

    with prep_tabs[1]:
        corr_rank = pd.DataFrame(bundle["corr_top"])
        top_features = corr_rank.head(12)["Feature"].tolist()
        heat_cols = top_features + ["Paddy yield(in Kg)"]
        corr_matrix = df[heat_cols].corr(numeric_only=True)
        fig = px.imshow(
            corr_matrix,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Correlation Heatmap — Top Numerical Predictors vs Paddy Yield",
        )
        fig.update_layout(height=720, margin=dict(l=10, r=10, t=60, b=20))
        st.plotly_chart(fig, use_container_width=True)

        corr_plot = corr_rank.head(15).sort_values("Correlation")
        fig2 = px.bar(
            corr_plot,
            x="Correlation",
            y="Feature",
            orientation="h",
            text="Correlation",
            title="Top 15 Numerical Correlations with Paddy Yield",
        )
        fig2.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig2.update_layout(height=500, yaxis_title="")
        st.plotly_chart(fig2, use_container_width=True)

    with prep_tabs[2]:
        selected_feature = st.selectbox("Choose a numerical feature", num_cols, key="distribution_feature")
        fig = px.histogram(
            df,
            x=selected_feature,
            color="Paddy Yield Category",
            color_discrete_map=CLASS_COLORS,
            barmode="overlay",
            opacity=0.68,
            nbins=35,
            title=f"Distribution of {selected_feature} by Paddy Yield Category",
        )
        fig.update_layout(height=520)
        st.plotly_chart(fig, use_container_width=True)

    with prep_tabs[3]:
        selected_cat = st.selectbox("Choose a categorical feature", cat_cols, key="categorical_feature")
        tab = pd.crosstab(df[selected_cat], df["Paddy Yield Category"], normalize="index") * 100
        tab = tab.reindex(columns=class_labels, fill_value=0).reset_index()
        long = tab.melt(id_vars=selected_cat, var_name="Paddy Yield Category", value_name="Percentage")
        fig = px.bar(
            long,
            x=selected_cat,
            y="Percentage",
            color="Paddy Yield Category",
            barmode="stack",
            color_discrete_map=CLASS_COLORS,
            category_orders={"Paddy Yield Category": class_labels},
            title=f"Paddy Yield Category Mix by {selected_cat}",
        )
        fig.update_layout(height=520, yaxis_range=[0, 100], yaxis_ticksuffix="%")
        st.plotly_chart(fig, use_container_width=True)

    with prep_tabs[4]:
        st.subheader("Before vs After StandardScaler (Training Data Only)")
        scale_feature = st.selectbox(
            "Feature to inspect",
            ["Trash(in bundles)", "Min temp_D1_D30", "Relative Humidity_D91_D120", "30DRain( in mm)"],
            key="scale_feature",
        )
        X_all = df[bundle["selected_features"]]
        y_all = df["Paddy Yield Category"]
        X_train_demo, _, _, _ = train_test_split(
            X_all, y_all, test_size=0.25, random_state=42, stratify=y_all
        )
        scaler = StandardScaler()
        scaled = scaler.fit_transform(X_train_demo[num_cols])
        scaled_df = pd.DataFrame(scaled, columns=num_cols, index=X_train_demo.index)
        before_col, after_col = st.columns(2)
        with before_col:
            fig = px.histogram(X_train_demo, x=scale_feature, nbins=35, title=f"{scale_feature} — Before")
            fig.update_layout(height=430)
            st.plotly_chart(fig, use_container_width=True)
        with after_col:
            fig = px.histogram(scaled_df, x=scale_feature, nbins=35, title=f"{scale_feature} — After")
            fig.update_layout(height=430)
            st.plotly_chart(fig, use_container_width=True)
        st.caption("The scaler is fitted on X_train only, then applied to the test set, matching the notebook's leakage-prevention approach.")

    with prep_tabs[5]:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Retained Predictors")
            retained = pd.DataFrame({"Selected Feature": bundle["selected_features"]})
            st.dataframe(retained, use_container_width=True, hide_index=True, height=430)
        with c2:
            st.subheader("Excluded Redundant Predictors")
            removed = pd.DataFrame({"Excluded Feature": bundle["excluded_features"]})
            st.dataframe(removed, use_container_width=True, hide_index=True, height=430)

        imp = pd.DataFrame(bundle["rf_feature_importance"]).head(15).sort_values("Importance")
        fig = px.bar(
            imp,
            x="Importance",
            y="Feature",
            orientation="h",
            text="Importance",
            title="Top 15 Random Forest Feature Importances",
        )
        fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig.update_layout(height=560, yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

with main_tabs[3]:
    st.markdown(
        """
        <div class="hero">
            <h1>Model Performance</h1>
            <p>Evaluation of Logistic Regression, KNN, Random Forest and ANN using the fixed train/test split and Macro metrics from your notebook workflow.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    score = pd.DataFrame(bundle["metrics"])
    score["5-Fold CV F1"] = score["Model"].map(
        {name: float(np.mean(values)) for name, values in bundle["cv_folds"].items()}
    )
    score = score[["Model", "Accuracy", "Precision", "Recall", "F1 Score", "5-Fold CV F1", "AUC", "Log Loss"]]

    st.subheader("Model Scorecard")
    best_test_model = score.loc[score["F1 Score"].idxmax(), "Model"]
    best_cv_model = score.loc[score["5-Fold CV F1"].idxmax(), "Model"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Best Test Macro F1", best_test_model, f"{score['F1 Score'].max():.4f}")
    c2.metric("Best CV Macro F1", best_cv_model, f"{score['5-Fold CV F1'].max():.4f}")
    c3.metric("Best Macro AUC", score.loc[score["AUC"].idxmax(), "Model"], f"{score['AUC'].max():.4f}")

    metric_cols = ["Accuracy", "Precision", "Recall", "F1 Score", "5-Fold CV F1", "AUC"]
    def style_scorecard(data):
        styles = pd.DataFrame("", index=data.index, columns=data.columns)
        for col in metric_cols:
            if col in data.columns:
                styles.loc[data[col].idxmax(), col] = "background-color:#dff3e5;font-weight:700;"
        styles.loc[data["Log Loss"].idxmin(), "Log Loss"] = "background-color:#dff3e5;font-weight:700;"
        return styles

    st.dataframe(
        score.style.apply(style_scorecard, axis=None).format({
            "Accuracy": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}",
            "F1 Score": "{:.4f}", "5-Fold CV F1": "{:.4f}", "AUC": "{:.4f}", "Log Loss": "{:.4f}"
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Performance Comparison")
    compare_long = score.melt(
        id_vars="Model",
        value_vars=["Accuracy", "Precision", "Recall", "F1 Score", "AUC"],
        var_name="Metric",
        value_name="Score",
    )
    fig = px.bar(
        compare_long,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        text=compare_long["Score"].map(lambda x: f"{x:.3f}"),
    )
    fig.update_layout(height=520, yaxis_range=[0, 1.05], legend_orientation="h", legend_y=-0.23)
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("K-Fold Cross-Validation")
    fold_rows = []
    for model_name, values in bundle["cv_folds"].items():
        for i, val in enumerate(values, 1):
            fold_rows.append({"Model": model_name, "Fold": f"Fold {i}", "Macro F1": val})
    fold_df = pd.DataFrame(fold_rows)
    fig = px.bar(fold_df, x="Fold", y="Macro F1", color="Model", barmode="group", text="Macro F1")
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(height=500, yaxis_range=[0, 1.05], legend_orientation="h", legend_y=-0.22)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Training Visualizations")
    viz_tabs = st.tabs([
        "Confusion Matrices", "ROC Curves", "PR Curves", "Feature Importance", "Learning Curve", "Tuning Evidence"
    ])

    with viz_tabs[0]:
        matrix_items = list(bundle["confusion_matrices"].items())
        for row_start in range(0, len(matrix_items), 2):
            cols = st.columns(2)
            for j, (name, matrix) in enumerate(matrix_items[row_start:row_start + 2]):
                with cols[j]:
                    cm = np.array(matrix)
                    fig = go.Figure(data=go.Heatmap(
                        z=cm,
                        x=class_labels,
                        y=class_labels,
                        colorscale="Blues",
                        showscale=False,
                        text=cm,
                        texttemplate="%{text}",
                        hovertemplate="Actual=%{y}<br>Predicted=%{x}<br>Count=%{z}<extra></extra>",
                    ))
                    fig.update_layout(
                        title=name,
                        height=390,
                        xaxis_title="Predicted",
                        yaxis_title="Actual",
                        yaxis_autorange="reversed",
                        margin=dict(l=60, r=20, t=55, b=50),
                    )
                    st.plotly_chart(fig, use_container_width=True)

    with viz_tabs[1]:
        roc_model = st.selectbox("Model", list(bundle["roc_data"].keys()), key="roc_model")
        fig = go.Figure()
        for class_name, vals in bundle["roc_data"][roc_model].items():
            fig.add_trace(go.Scatter(
                x=vals["fpr"], y=vals["tpr"], mode="lines",
                name=f"{class_name} (AUC={vals['auc']:.3f})"
            ))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random classifier", line=dict(dash="dash")))
        fig.update_layout(
            title=f"One-vs-Rest ROC Curves — {roc_model}",
            xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
            height=570, xaxis_range=[0, 1], yaxis_range=[0, 1.02]
        )
        st.plotly_chart(fig, use_container_width=True)

    with viz_tabs[2]:
        pr_model = st.selectbox("Model", list(bundle["pr_data"].keys()), key="pr_model")
        fig = go.Figure()
        for class_name, vals in bundle["pr_data"][pr_model].items():
            fig.add_trace(go.Scatter(
                x=vals["recall"], y=vals["precision"], mode="lines",
                name=f"{class_name} (AP={vals['ap']:.3f})"
            ))
        fig.update_layout(
            title=f"One-vs-Rest Precision–Recall Curves — {pr_model}",
            xaxis_title="Recall", yaxis_title="Precision", height=570,
            xaxis_range=[0, 1], yaxis_range=[0, 1.02]
        )
        st.plotly_chart(fig, use_container_width=True)

    with viz_tabs[3]:
        imp_type = st.radio("Importance view", ["Random Forest", "Logistic Regression"], horizontal=True)
        source = bundle["rf_feature_importance"] if imp_type == "Random Forest" else bundle["lr_feature_importance"]
        imp = pd.DataFrame(source).head(15).sort_values("Importance")
        fig = px.bar(imp, x="Importance", y="Feature", orientation="h", text="Importance",
                     title=f"Top 15 Feature Importances — {imp_type}")
        fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig.update_layout(height=600, yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with viz_tabs[4]:
        lc = bundle["rf_learning_curve"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=lc["train_sizes"], y=lc["train_mean"], mode="lines+markers", name="Training Macro F1"
        ))
        fig.add_trace(go.Scatter(
            x=lc["train_sizes"], y=lc["val_mean"], mode="lines+markers", name="Validation Macro F1"
        ))
        fig.update_layout(
            title="Random Forest Learning Curve",
            xaxis_title="Number of Training Records", yaxis_title="Macro F1",
            yaxis_range=[0, 1.05], height=550
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("The learning curve is precomputed from 5-fold stratified cross-validation to keep the deployed app fast.")

    with viz_tabs[5]:
        st.subheader("Best Hyperparameters")
        params_rows = []
        for model_name, params in bundle["best_params"].items():
            params_rows.append({"Model": model_name, "Best Parameters": str(params)})
        st.dataframe(pd.DataFrame(params_rows), use_container_width=True, hide_index=True)

        st.subheader("Random Forest Tuning Evidence")
        tuning = pd.DataFrame(bundle["rf_tuning_evidence"])
        st.dataframe(tuning, use_container_width=True, hide_index=True)
        fig = px.bar(
            tuning.sort_values("mean_cv_f1"),
            x="mean_cv_f1", y=tuning["Candidate"].astype(str), orientation="h",
            text="mean_cv_f1", title="Top Random Forest GridSearchCV Candidates"
        )
        fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig.update_layout(height=420, xaxis_range=[0.90, 1.0], yaxis_title="Candidate")
        st.plotly_chart(fig, use_container_width=True)

with main_tabs[4]:
    st.markdown(
        """
        <div class="hero">
            <h1>About the Project</h1>
            <p>A Streamlit interface built directly around the preprocessing and modelling structure in the uploaded paddy notebook.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Project Objective")
    st.write(
        "Given the characteristics and conditions of a farm, predict the expected level of total paddy production. "
        "The target is a four-class classification problem: Low, Moderate, High and Very High."
    )

    summary = bundle["data_summary"]
    cols = st.columns(5)
    cols[0].metric("Original Rows", f"{summary['rows_original']:,}")
    cols[1].metric("Duplicates Removed", f"{summary['duplicates_removed']:,}")
    cols[2].metric("Cleaned Rows", f"{summary['rows_cleaned']:,}")
    cols[3].metric("Selected Predictors", summary["selected_features"])
    cols[4].metric("Prepared Features", summary["prepared_features"])

    st.subheader("Target Definition")
    target_table = pd.DataFrame({
        "Class": class_labels,
        "Paddy Yield Range": [yield_ranges[c] for c in class_labels],
        "Cleaned Dataset Count": [bundle["class_counts"][c] for c in class_labels],
    })
    st.dataframe(target_table, use_container_width=True, hide_index=True)

    st.subheader("Preprocessing Flow")
    st.markdown(
        """
        1. Remove duplicate records.  
        2. Confirm missing values and inspect outliers.  
        3. Exclude the 11 highly redundant hectare-related variables used in the notebook.  
        4. Create four paddy-yield categories.  
        5. Use a stratified 75/25 train-test split with `random_state=42`.  
        6. Fit `StandardScaler` to numerical training variables only.  
        7. One-hot encode categorical variables with unknown-category handling.  
        8. Train and evaluate Logistic Regression, KNN, Random Forest and ANN.
        """
    )

    st.info(
        "The app intentionally keeps the same 33 predictors and target definition as the uploaded notebook. "
        "It does not add RFECV, SMOTE, threshold optimisation or SHAP because those are not part of the supplied notebook workflow."
    )
