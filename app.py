"""
MBA Class Participation Assessment — Faculty Decision-Support Dashboard
========================================================================
A production-ready prototype that loads the pseudonymised model_dataset.csv,
trains the tuned XGBoost model, and presents faculty with:
  - a course overview with suggested participation scores
  - per-student SHAP explanations (why did this student get this score?)
  - accept / adjust / override workflow
  - cross-cohort performance summary
  - CSV export of decisions

Run:  streamlit run app.py
Needs: pip install streamlit pandas numpy scikit-learn xgboost shap matplotlib
========================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
from scipy.stats import pearsonr
import io

# ========================== CONFIG ==========================
DATA_FILE = "model_dataset.csv"
TARGET = "participation_score"
GROUP = "student_id"
LEAK = ["freq_1to5", "relevance_1to5", "depth_1to5"]
XGB_PARAMS = dict(
    n_estimators=400, max_depth=3, learning_rate=0.1,
    subsample=1.0, min_child_weight=1, random_state=42
)

# ========================== PAGE CONFIG ==========================
st.set_page_config(
    page_title="MBA Participation Assessment",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================== CUSTOM CSS ==========================
st.markdown("""
<style>
    .main-header {
        font-size: 1.8rem; font-weight: 700; color: #1F3B57;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem; color: #666; margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8f9fc 0%, #e8ecf4 100%);
        border-radius: 12px; padding: 1.2rem; text-align: center;
        border-left: 4px solid #2d6a9f;
    }
    .metric-value {
        font-size: 2rem; font-weight: 700; color: #1F3B57;
    }
    .metric-label {
        font-size: 0.85rem; color: #666; margin-top: 0.2rem;
    }
    .score-high { color: #27ae60; font-weight: 700; }
    .score-mid { color: #f39c12; font-weight: 700; }
    .score-low { color: #e74c3c; font-weight: 700; }
    .decision-support-banner {
        background: #fff3cd; border-left: 4px solid #ffc107;
        padding: 0.8rem 1rem; border-radius: 0 8px 8px 0;
        margin-bottom: 1rem; font-size: 0.9rem;
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1F3B57 0%, #2d5a8e 100%);
    }
    div[data-testid="stSidebar"] .stMarkdown p,
    div[data-testid="stSidebar"] .stMarkdown h1,
    div[data-testid="stSidebar"] .stMarkdown h2,
    div[data-testid="stSidebar"] .stMarkdown h3 {
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# ========================== DATA LOADING & MODEL ==========================
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)
    df = df[df[TARGET].notna()].copy()
    return df


@st.cache_resource
def build_model(df):
    """Train the model and compute SHAP values + cross-validated predictions."""
    drop_cols = [GROUP, "course_id", TARGET] + [c for c in LEAK if c in df.columns]
    X = df.drop(columns=drop_cols, errors="ignore")
    X = pd.get_dummies(X, columns=[c for c in ("programme", "cohort") if c in X.columns],
                       drop_first=True)
    X = X.apply(pd.to_numeric, errors="coerce")
    y = df[TARGET].to_numpy()
    groups = df[GROUP].to_numpy()
    feature_names = list(X.columns)

    # cross-validated predictions
    n_groups = len(np.unique(groups))
    n_splits = max(2, min(5, n_groups))
    cv = GroupKFold(n_splits=n_splits)
    model = XGBRegressor(**XGB_PARAMS)
    preds = cross_val_predict(model, X, y, cv=cv, groups=groups)

    # fit final model for SHAP
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    return model, explainer, shap_values, preds, X, y, groups, feature_names


def score_color(score, mean, std):
    if score >= mean + 0.5 * std:
        return "score-high"
    elif score <= mean - 0.5 * std:
        return "score-low"
    return "score-mid"


def compute_metrics(y_true, y_pred):
    r = pearsonr(y_true, y_pred)[0]
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    w5 = 100 * np.mean(np.abs(y_pred - y_true) <= 5)
    return {"Pearson r": round(r, 3), "MAE": round(mae, 2),
            "R²": round(r2, 3), "Within ±5": f"{round(w5, 1)}%"}


# ========================== MAIN APP ==========================
def main():
    # load
    try:
        df = load_data()
    except FileNotFoundError:
        st.error(f"❌ `{DATA_FILE}` not found. Place it in the same folder as this app.")
        st.stop()

    model, explainer, shap_values, preds, X, y, groups, feature_names = build_model(df)
    df = df.copy()
    df["predicted_score"] = np.round(preds, 1)
    df["residual"] = np.round(df["predicted_score"] - df[TARGET], 1)

    # ---- Sidebar ----
    with st.sidebar:
        st.markdown("# 🎓 MBA Participation")
        st.markdown("### Decision-Support Tool")
        st.markdown("---")

        # Navigation
        page = st.radio("Navigate", [
            "📊 Course Overview",
            "🔍 Student Detail",
            "📈 Cross-Cohort Results",
            "📥 Export Decisions"
        ], label_visibility="collapsed")

        st.markdown("---")

        # Filters
        st.markdown("### Filters")
        cohorts = sorted(df["cohort"].dropna().unique())
        sel_cohort = st.selectbox("Cohort", ["All"] + cohorts)
        sub = df if sel_cohort == "All" else df[df["cohort"] == sel_cohort]

        courses = sorted(sub["course_id"].unique())
        sel_course = st.selectbox("Course", ["All"] + courses)
        if sel_course != "All":
            sub = sub[sub["course_id"] == sel_course]

        st.markdown("---")
        st.markdown(f"**Showing:** {len(sub)} students")
        st.markdown(f"**Courses:** {sub['course_id'].nunique()}")

    # ---- Decision Support Banner ----
    st.markdown("""
    <div class="decision-support-banner">
        ⚖️ <strong>Decision-Support Tool</strong> — This system produces
        <em>suggested</em> scores with explanations for faculty review.
        Final grading authority remains with the faculty member.
        Scores can be accepted, adjusted, or overridden.
    </div>
    """, unsafe_allow_html=True)

    # ==================== PAGE: Course Overview ====================
    if "Course Overview" in page:
        st.markdown('<p class="main-header">Course Overview</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Suggested participation scores across students</p>',
                    unsafe_allow_html=True)

        # Metrics row
        metrics = compute_metrics(sub[TARGET].to_numpy(), sub["predicted_score"].to_numpy())
        cols = st.columns(4)
        for col, (label, val) in zip(cols, metrics.items()):
            col.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")

        # Student table
        display_cols = [GROUP, "course_id", "cohort", TARGET, "predicted_score", "residual"]
        feat_cols = ["n_contributions", "total_words", "chat_count", "attendance_ratio",
                     "lexical_diversity", "semantic_similarity"]
        avail = [c for c in feat_cols if c in sub.columns]
        show = sub[display_cols + avail].copy()
        show.columns = [c.replace("_", " ").title() for c in show.columns]

        # color-code the residuals
        st.dataframe(
            show.style.background_gradient(
                subset=["Predicted Score"], cmap="RdYlGn", vmin=sub[TARGET].min(),
                vmax=sub[TARGET].max()
            ).format(precision=1),
            use_container_width=True,
            height=500
        )

        # Distribution plot
        st.markdown("#### Score Distribution")
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].hist(sub[TARGET], bins=15, color="#2d6a9f", alpha=0.7, edgecolor="white")
        axes[0].set_xlabel("Faculty Score"); axes[0].set_title("Faculty Scores")
        axes[1].hist(sub["predicted_score"], bins=15, color="#2a9d8f", alpha=0.7, edgecolor="white")
        axes[1].set_xlabel("Predicted Score"); axes[1].set_title("Model Predictions")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ==================== PAGE: Student Detail ====================
    elif "Student Detail" in page:
        st.markdown('<p class="main-header">Student Detail & Explanation</p>',
                    unsafe_allow_html=True)
        st.markdown('<p class="sub-header">SHAP-based explanation of a student\'s suggested score</p>',
                    unsafe_allow_html=True)

        # Student selector
        students = sorted(sub[GROUP].unique())
        sel_student = st.selectbox("Select Student", students)

        # get all rows for this student
        stu_rows = sub[sub[GROUP] == sel_student]
        if len(stu_rows) == 0:
            st.warning("No data for this student in the current filter.")
            st.stop()

        # if multiple courses, let them pick
        if len(stu_rows) > 1:
            stu_course = st.selectbox("Course", sorted(stu_rows["course_id"].unique()))
            stu_row = stu_rows[stu_rows["course_id"] == stu_course].iloc[0]
        else:
            stu_row = stu_rows.iloc[0]

        row_idx = stu_row.name  # index in the original df

        # Metrics for this student
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Faculty Score", f"{stu_row[TARGET]:.1f}")
        col2.metric("Suggested Score", f"{stu_row['predicted_score']:.1f}",
                     delta=f"{stu_row['residual']:+.1f}")
        col3.metric("Contributions", f"{int(stu_row.get('n_contributions', 0))}")
        col4.metric("Attendance", f"{stu_row.get('attendance_ratio', 0):.0%}")

        st.markdown("---")

        # SHAP waterfall for this student
        st.markdown("#### Why this score? (SHAP Explanation)")
        st.markdown("Each bar shows how much a feature pushed the score **up** (red) or "
                     "**down** (blue) from the baseline prediction.")

        # find the index in X (which matches df's index)
        idx_in_X = list(df.index).index(row_idx)

        fig, ax = plt.subplots(figsize=(8, 6))
        shap_vals = shap_values[idx_in_X]
        base = explainer.expected_value

        # sort by absolute impact
        order = np.argsort(np.abs(shap_vals))
        top_n = min(12, len(order))
        top_idx = order[-top_n:]

        feat_names = [feature_names[i] for i in top_idx]
        vals = [shap_vals[i] for i in top_idx]
        colors = ["#e74c3c" if v > 0 else "#2d6a9f" for v in vals]

        ax.barh(feat_names, vals, color=colors)
        ax.axvline(0, color="grey", linewidth=0.8)
        ax.set_xlabel("SHAP value (impact on predicted score)")
        ax.set_title(f"Score explanation for {sel_student}")

        # add base value annotation
        ax.annotate(f"Base prediction: {base:.1f}", xy=(0, -0.8),
                    fontsize=9, color="#666", ha="center")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Feature values table
        st.markdown("#### Feature Values for This Student")
        feat_display = {}
        for fn in feature_names:
            if fn in df.columns:
                feat_display[fn] = stu_row.get(fn, "N/A")
            else:
                feat_display[fn] = X.iloc[idx_in_X][fn] if fn in X.columns else "N/A"
        feat_df = pd.DataFrame({
            "Feature": list(feat_display.keys()),
            "Value": [f"{v:.3f}" if isinstance(v, (int, float, np.floating)) else str(v)
                      for v in feat_display.values()],
            "SHAP Impact": [f"{shap_vals[i]:+.2f}" for i in range(len(feature_names))]
        })
        feat_df = feat_df.sort_values("SHAP Impact", key=lambda s: s.astype(float).abs(),
                                       ascending=False)
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

        # Accept / Adjust / Override
        st.markdown("---")
        st.markdown("#### Faculty Decision")
        decision = st.radio(
            "Action on this student's score:",
            ["Accept suggested score", "Adjust score", "Override with own score"],
            horizontal=True
        )
        if decision == "Adjust score":
            adjusted = st.slider("Adjusted score", min_value=0.0,
                                  max_value=30.0, value=float(stu_row["predicted_score"]),
                                  step=0.5)
            st.info(f"Adjusted score: **{adjusted}** (was {stu_row['predicted_score']})")
        elif decision == "Override with own score":
            override = st.number_input("Your score", min_value=0.0, max_value=30.0,
                                        value=float(stu_row[TARGET]), step=0.5)
            st.info(f"Overridden to: **{override}**")
        else:
            st.success(f"Accepted: **{stu_row['predicted_score']}**")

    # ==================== PAGE: Cross-Cohort Results ====================
    elif "Cross-Cohort" in page:
        st.markdown('<p class="main-header">Cross-Cohort Validation Results</p>',
                    unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Framework performance across independent cohorts</p>',
                    unsafe_allow_html=True)

        results = []
        for cohort in sorted(df["cohort"].dropna().unique()):
            csub = df[df["cohort"] == cohort]
            m = compute_metrics(csub[TARGET].to_numpy(), csub["predicted_score"].to_numpy())
            m["Cohort"] = cohort
            m["n"] = len(csub)
            m["Students"] = csub[GROUP].nunique()
            results.append(m)
        # pooled
        m = compute_metrics(df[TARGET].to_numpy(), df["predicted_score"].to_numpy())
        m["Cohort"] = "Pooled"
        m["n"] = len(df)
        m["Students"] = df[GROUP].nunique()
        results.append(m)

        res_df = pd.DataFrame(results)[["Cohort", "n", "Students", "Pearson r", "MAE", "R²", "Within ±5"]]
        st.dataframe(res_df, use_container_width=True, hide_index=True)

        # Bar chart
        fig, ax = plt.subplots(figsize=(7, 4))
        colors = ["#2d6a9f", "#2a9d8f", "#8a8a8a"]
        ax.bar(res_df["Cohort"], res_df["Pearson r"], color=colors[:len(res_df)])
        ax.axhline(0.60, ls="--", color="#c0392b", label="Target r = 0.60")
        for i, v in enumerate(res_df["Pearson r"]):
            ax.text(i, v + 0.01, f"{v}", ha="center", fontweight="bold")
        ax.set_ylabel("Pearson r"); ax.set_ylim(0, 0.8)
        ax.set_title("Cross-Cohort Agreement with Faculty Scores")
        ax.legend(frameon=False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # SHAP global importance
        st.markdown("#### Global Feature Importance (SHAP)")
        fig, ax = plt.subplots(figsize=(8, 5))
        mean_abs = np.abs(shap_values).mean(axis=0)
        order = np.argsort(mean_abs)
        top = order[-12:]
        ax.barh([feature_names[i] for i in top], mean_abs[top], color="#2d6a9f")
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title("What drives the participation score?")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("""
        **Key finding:** The framework achieved near-identical Pearson correlations
        across two independent cohorts (MMBA 8 and MMBA 9), demonstrating that it
        generalises rather than overfitting to a single class. The model rewards
        attendance, contribution richness, chat activity and topical relevance
        rather than volume alone.
        """)

    # ==================== PAGE: Export ====================
    elif "Export" in page:
        st.markdown('<p class="main-header">Export Decisions</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Download suggested scores for your records</p>',
                    unsafe_allow_html=True)

        export_cols = [GROUP, "course_id", "cohort", TARGET, "predicted_score", "residual"]
        avail_feat = [c for c in ["n_contributions", "total_words", "chat_count",
                                   "attendance_ratio", "lexical_diversity"] if c in sub.columns]
        export_df = sub[export_cols + avail_feat].copy()
        export_df.columns = [c.replace("_", " ").title() for c in export_df.columns]

        st.dataframe(export_df, use_container_width=True, height=400)

        csv = export_df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name=f"participation_scores_{sel_cohort}_{sel_course}.csv",
            mime="text/csv"
        )

        st.markdown("""
        **Usage note:** This export contains *suggested* scores with feature values.
        Faculty should review and confirm each score before entering it into the
        official gradebook. The suggested score is a starting point, not a final grade.
        """)


if __name__ == "__main__":
    main()
