import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓"
)

st.title("🎓 Student Performance Prediction System")
st.write("AI-Based Student Performance Analysis")

data = {
    "Study_Hours": [1, 2, 3, 4, 5, 6, 2, 4, 5, 3],
    "Attendance": [50, 60, 70, 80, 90, 95, 65, 85, 92, 75],
    "Previous_Marks": [40, 45, 55, 65, 75, 85, 50, 70, 80, 60],
    "Final_Marks": [42, 48, 58, 68, 78, 88, 52, 73, 83, 63]
}

df = pd.DataFrame(data)

X = df[["Study_Hours", "Attendance", "Previous_Marks"]]
y = df["Final_Marks"]

model = LinearRegression()
model.fit(X, y)

st.header("Enter Student Details")

study_hours = st.number_input(
    "Daily Study Hours", min_value=0.0,
    max_value=24.0, value=3.0
)

attendance = st.number_input(
    "Attendance (%)", min_value=0.0,
    max_value=100.0, value=85.0
)

previous_marks = st.number_input(
    "Previous Marks", min_value=0.0,
    max_value=100.0, value=70.0
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