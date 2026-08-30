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
    data = pd.read_csv(DATA_PATH)
    data.columns = data.columns.str.strip()
    data = data.drop_duplicates().reset_index(drop=True)
    data["Paddy Yield Category"] = pd.cut(
        data["Paddy yield(in Kg)"],
        bins=[0, 10000, 20000, 30000, 40000],
        labels=["Low", "Moderate", "High", "Very High"],
        right=False,
    )
    return data


try:
    bundle = load_bundle()
    df = load_data()
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
main_tabs = st.tabs(["Prediction", "Data Exploration", "Model Performance", "About"])

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
            <h1>Data Exploration</h1>
            <p>Preprocessing evidence built from the same cleaned paddy dataset used in the notebook.</p>
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

with main_tabs[2]:
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

with main_tabs[3]:
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
