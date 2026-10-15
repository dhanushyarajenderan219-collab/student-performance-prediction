import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(
    page_title="Student Performance AI",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Performance Analytics & Prediction")
st.caption("Machine Learning | Academic Analytics | Student Support")

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose Page",
    ["Dashboard", "Predict Performance", "Model Comparison", "Student Analytics"]
)

st.sidebar.info(
    "This application is an educational prototype. "
    "Predictions are estimates, not guarantees."
)

# --------------------------------------------------
# SAMPLE DATA
# --------------------------------------------------

sample_data = {
    "Study_Hours": [1, 2, 3, 4, 5, 6, 2, 4, 5, 3,
                    7, 6, 1, 8, 5, 4, 7, 2, 6, 3],
    "Attendance": [50, 60, 70, 80, 90, 95, 65, 85, 92, 75,
                   96, 88, 45, 98, 82, 72, 93, 55, 90, 68],
    "Previous_Marks": [40, 45, 55, 65, 75, 85, 50, 70, 80, 60,
                       88, 78, 35, 92, 72, 58, 86, 42, 82, 53],
    "Final_Marks": [42, 48, 58, 68, 78, 88, 52, 73, 83, 63,
                    90, 81, 38, 95, 75, 61, 89, 45, 85, 56]
}

sample_df = pd.DataFrame(sample_data)

st.sidebar.subheader("Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload student CSV (optional)",
    type=["csv"]
)

required_columns = [
    "Study_Hours",
    "Attendance",
    "Previous_Marks",
    "Final_Marks"
]

if uploaded_file is not None:
    try:
        uploaded_df = pd.read_csv(uploaded_file)
        missing = [
            col for col in required_columns
            if col not in uploaded_df.columns
        ]

        if missing:
            st.sidebar.error(
                "Missing columns: " + ", ".join(missing)
            )
            df = sample_df.copy()
        elif len(uploaded_df) < 5:
            st.sidebar.error(
                "Please upload at least 5 student records."
            )
            df = sample_df.copy()
        else:
            df = uploaded_df[required_columns].copy()
            st.sidebar.success("CSV uploaded successfully!")
    except Exception:
        st.sidebar.error("Unable to read this CSV file.")
        df = sample_df.copy()
else:
    df = sample_df.copy()

# Convert required columns to numbers and clean invalid rows
for col in required_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=["Final_Marks"])
df = df[
    df["Final_Marks"].between(0, 100)
    & df["Study_Hours"].between(0, 24)
    & df["Attendance"].between(0, 100)
    & df["Previous_Marks"].between(0, 100)
].copy()

if len(df) < 5:
    st.error(
        "Not enough valid records. Please upload at least "
        "5 valid student records."
    )
    st.stop()

FEATURES = ["Study_Hours", "Attendance", "Previous_Marks"]
TARGET = "Final_Marks"

X = df[FEATURES]
y = df[TARGET]

# --------------------------------------------------
# MODEL TRAINING
# --------------------------------------------------

models = {
    "Linear Regression": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LinearRegression()
    ),
    "Decision Tree": make_pipeline(
        SimpleImputer(strategy="median"),
        DecisionTreeRegressor(max_depth=4, random_state=42)
    ),
    "Random Forest": make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
    )
}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

results = []
trained_models = {}

for name, estimator in models.items():
    estimator.fit(X_train, y_train)
    predicted = estimator.predict(X_test)

    mae = mean_absolute_error(y_test, predicted)
    r2 = r2_score(y_test, predicted) if len(y_test) > 1 else np.nan

    results.append({
        "Model": name,
        "MAE": round(mae, 2),
        "R2 Score": round(r2, 3) if np.isfinite(r2) else np.nan
    })

    trained_models[name] = estimator

results_df = pd.DataFrame(results).sort_values("MAE")
best_name = results_df.iloc[0]["Model"]

# Retrain models on all available records for the prediction demo
for estimator in trained_models.values():
    estimator.fit(X, y)

best_model = trained_models[best_name]

# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

if page == "Dashboard":
    st.header("📊 Student Overview")

    average_marks = df[TARGET].mean()
    average_attendance = df["Attendance"].mean()
    students_needing_support = int(
        ((df[TARGET] < 50) | (df["Attendance"] < 75)).sum()
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Student Records", len(df))
    c2.metric("Average Marks", f"{average_marks:.1f}/100")
    c3.metric("Average Attendance", f"{average_attendance:.1f}%")
    c4.metric("May Need Support", students_needing_support)

    st.subheader("Marks Distribution")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(df[TARGET], bins=8, edgecolor="black")
    ax.set_xlabel("Final Marks")
    ax.set_ylabel("Number of Students")
    ax.set_title("Distribution of Student Marks")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Student Dataset")
    st.dataframe(df, use_container_width=True)

    st.download_button(
        "⬇️ Download Dataset",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="student_dataset.csv",
        mime="text/csv"
    )

# --------------------------------------------------
# PERFORMANCE PREDICTION
# --------------------------------------------------

elif page == "Predict Performance":
    st.header("🤖 Predict Student Performance")
    st.write("Enter the student's details below.")

    col1, col2 = st.columns(2)

    with col1:
        study_hours = st.number_input(
            "Daily Study Hours",
            min_value=0.0,
            max_value=24.0,
            value=4.0,
            step=0.5
        )

        attendance = st.number_input(
            "Attendance (%)",
            min_value=0.0,
            max_value=100.0,
            value=80.0,
            step=1.0
        )

    with col2:
        previous_marks = st.number_input(
            "Previous Marks",
            min_value=0.0,
            max_value=100.0,
            value=65.0,
            step=1.0
        )

        selected_model = st.selectbox(
            "Select ML Model",
            list(trained_models.keys()),
            index=list(trained_models.keys()).index(best_name)
        )

    if st.button("Predict Performance", type="primary"):
        student = pd.DataFrame([{
            "Study_Hours": study_hours,
            "Attendance": attendance,
            "Previous_Marks": previous_marks
        }])

        prediction = trained_models[selected_model].predict(student)[0]
        prediction = float(np.clip(prediction, 0, 100))

        st.subheader("Prediction Result")
        st.metric("Estimated Final Marks", f"{prediction:.1f}/100")

        if prediction >= 75:
            st.success("Estimated performance category: High")
        elif prediction >= 50:
            st.info("Estimated performance category: Moderate")
        else:
            st.warning(
                "Estimated performance category: Needs additional support"
            )

        st.subheader("Suggested Academic Support")

        suggestions = []

        if study_hours < 3:
            suggestions.append(
                "Consider a regular study timetable with manageable daily goals."
            )

        if attendance < 75:
            suggestions.append(
                "Improve class attendance where possible and discuss missed lessons with a teacher."
            )

        if previous_marks < 50:
            suggestions.append(
                "Review foundational topics and practise questions with teacher support."
            )

        if not suggestions:
            suggestions.append(
                "Continue consistent study habits and review difficult topics regularly."
            )

        for suggestion in suggestions:
            st.write("•", suggestion)

        st.caption(
            "Demonstration estimate only. This model uses a small dataset "
            "and must not be used to make high-stakes decisions about students."
        )

# --------------------------------------------------
# MODEL COMPARISON
# --------------------------------------------------

elif page == "Model Comparison":
    st.header("🧠 Machine Learning Model Comparison")
    st.write(
        "Models are evaluated on a held-out test set using the selected dataset."
    )

    st.dataframe(results_df, use_container_width=True)

    st.subheader("Mean Absolute Error (MAE)")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(results_df["Model"], results_df["MAE"])
    ax.set_ylabel("MAE (marks)")
    ax.set_title("Lower MAE generally means smaller prediction errors")
    ax.tick_params(axis="x", rotation=15)
    st.pyplot(fig)
    plt.close(fig)

    st.info(
        f"Lowest test-set MAE in this run: {best_name}. "
        "With very small datasets, scores can vary substantially and "
        "do not establish real-world accuracy."
    )

# --------------------------------------------------
# STUDENT ANALYTICS
# --------------------------------------------------

elif page == "Student Analytics":
    st.header("📈 Student Performance Analytics")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Study Hours vs Final Marks")

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(df["Study_Hours"], df["Final_Marks"])
        ax.set_xlabel("Daily Study Hours")
        ax.set_ylabel("Final Marks")
        ax.set_title("Study Hours and Marks")
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.subheader("Attendance vs Final Marks")

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(df["Attendance"], df["Final_Marks"])
        ax.set_xlabel("Attendance (%)")
        ax.set_ylabel("Final Marks")
        ax.set_title("Attendance and Marks")
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Correlation Between Variables")

    fig, ax = plt.subplots(figsize=(7, 4))
    corr = df[FEATURES + [TARGET]].corr(numeric_only=True)
    image = ax.imshow(corr, aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)
    ax.set_title("Correlation Matrix")
    fig.colorbar(image, ax=ax)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Students Who May Benefit From Support")

    support_df = df[
        (df["Final_Marks"] < 50) | (df["Attendance"] < 75)
    ].copy()

    if support_df.empty:
        st.success(
            "No records meet these simple support flags in the current dataset."
        )
    else:
        st.write(
            "These flags are only prompts for supportive review, "
            "not labels about a student's ability."
        )
        st.dataframe(support_df, use_container_width=True)

        st.download_button(
            "⬇️ Download Support Review CSV",
            data=support_df.to_csv(index=False).encode("utf-8"),
            file_name="student_support_review.csv",
            mime="text/csv"
        )

st.divider()
st.caption(
    "Built with Python, Streamlit, pandas, Matplotlib and scikit-learn."
)
if st.button("Predict Performance"):
    student = pd.DataFrame([{
        "Study_Hours": study_hours,
        "Attendance": attendance,
        "Previous_Marks": previous_marks
    }])

    result = model.predict(student)[0]
    result = max(0, min(100, result))

    st.success(f"Predicted Marks: {result:.2f}/100")

    st.info(
        "This is a demonstration using a small sample dataset. "
        "The prediction is not a reliable measure of actual performance."
    )

st.header("Student Performance Graphs")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].scatter(df["Study_Hours"], df["Final_Marks"])
axes[0].set_title("Study Hours vs Marks")
axes[0].set_xlabel("Study Hours")
axes[0].set_ylabel("Final Marks")

axes[1].scatter(df["Attendance"], df["Final_Marks"])
axes[1].set_title("Attendance vs Marks")
axes[1].set_xlabel("Attendance (%)")
axes[1].set_ylabel("Final Marks")

plt.tight_layout()
st.pyplot(fig)
plt.close(fig)