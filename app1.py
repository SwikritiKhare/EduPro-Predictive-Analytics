import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------

st.set_page_config(
    page_title="EduPro Predictive Analytics Dashboard",
    layout="wide"
)

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

file_path = "EduPro Online Platform.xlsx"

df_courses = pd.read_excel(
    file_path,
    sheet_name="Courses"
)

df_teachers = pd.read_excel(
    file_path,
    sheet_name="Teachers"
)

df_transactions = pd.read_excel(
    file_path,
    sheet_name="Transactions"
)

# ---------------------------------------------------
# MERGE DATASETS
# ---------------------------------------------------

merged = pd.merge(
    df_courses,
    df_transactions,
    on="CourseID"
)

merged = pd.merge(
    merged,
    df_teachers,
    on="TeacherID"
)

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

model = pickle.load(
    open("rf_model.pkl", "rb")
)

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.title("🎓 EduPro Predictive Analytics Dashboard")

st.markdown("""
This dashboard predicts course enrollment demand and provides
business insights for pricing, course planning, and instructor analysis.
""")

# ---------------------------------------------------
# KPI SECTION
# ---------------------------------------------------

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Courses",
    len(df_courses)
)

col2.metric(
    "Average Course Rating",
    round(merged['CourseRating'].mean(), 2)
)

col3.metric(
    "Total Revenue",
    f"${int(merged['Amount'].sum())}"
)

st.divider()

# ---------------------------------------------------
# SIDEBAR INPUTS
# ---------------------------------------------------

st.sidebar.header("📌 Enter Course Details")

course_category = st.sidebar.selectbox(
    "Course Category",
    sorted(merged['CourseCategory'].unique())
)

course_type = st.sidebar.selectbox(
    "Course Type",
    sorted(merged['CourseType'].unique())
)

course_level = st.sidebar.selectbox(
    "Course Level",
    sorted(merged['CourseLevel'].unique())
)

course_price = st.sidebar.slider(
    "Course Price",
    0,
    1000,
    200
)

course_duration = st.sidebar.slider(
    "Course Duration",
    1,
    50,
    10
)

course_rating = st.sidebar.slider(
    "Course Rating",
    1.0,
    5.0,
    4.0
)

teacher_rating = st.sidebar.slider(
    "Teacher Rating",
    1.0,
    5.0,
    4.5
)

years_experience = st.sidebar.slider(
    "Years Of Experience",
    0,
    25,
    5
)

# ---------------------------------------------------
# ENCODE INPUTS
# ---------------------------------------------------

category_mapping = {
    value: index
    for index, value in enumerate(
        sorted(merged['CourseCategory'].unique())
    )
}

type_mapping = {
    value: index
    for index, value in enumerate(
        sorted(merged['CourseType'].unique())
    )
}

level_mapping = {
    value: index
    for index, value in enumerate(
        sorted(merged['CourseLevel'].unique())
    )
}

input_data = pd.DataFrame({
    'CourseCategory': [category_mapping[course_category]],
    'CourseType': [type_mapping[course_type]],
    'CourseLevel': [level_mapping[course_level]],
    'CoursePrice': [course_price],
    'CourseDuration': [course_duration],
    'CourseRating': [course_rating],
    'TeacherRating': [teacher_rating],
    'YearsOfExperience': [years_experience]
})

# ---------------------------------------------------
# DISPLAY INPUTS
# ---------------------------------------------------

st.subheader("📋 Selected Inputs")

st.dataframe(input_data)

# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------

if st.button("🚀 Predict Enrollment"):

    prediction = model.predict(input_data)

    st.success(
        f"🎯 Predicted Enrollment Count: {int(prediction[0])} Students"
    )

st.divider()

# ---------------------------------------------------
# TABS
# ---------------------------------------------------

tab1, tab2, tab3 = st.tabs([
    "📊 Analytics",
    "📈 Visualizations",
    "💡 Business Insights"
])

# ---------------------------------------------------
# TAB 1 - ANALYTICS
# ---------------------------------------------------

with tab1:

    st.subheader("Course Category Revenue")

    category_revenue = merged.groupby(
        'CourseCategory'
    )['Amount'].sum().reset_index()

    st.dataframe(category_revenue)

    st.bar_chart(
        category_revenue.set_index('CourseCategory')
    )

# ---------------------------------------------------
# TAB 2 - VISUALIZATIONS
# ---------------------------------------------------

with tab2:

    st.subheader("Feature Visualization")

    fig, ax = plt.subplots(figsize=(8, 4))

    features = [
        'Price',
        'Duration',
        'Course Rating',
        'Teacher Rating',
        'Experience'
    ]

    values = [
        course_price,
        course_duration,
        course_rating,
        teacher_rating,
        years_experience
    ]

    sns.barplot(
        x=features,
        y=values,
        palette='viridis',
        ax=ax
    )

    plt.xticks(rotation=10)

    st.pyplot(fig)

    st.subheader("Course Rating Distribution")

    fig2, ax2 = plt.subplots(figsize=(8, 4))

    sns.histplot(
        merged['CourseRating'],
        bins=10,
        kde=True,
        ax=ax2
    )

    st.pyplot(fig2)

# ---------------------------------------------------
# TAB 3 - BUSINESS INSIGHTS
# ---------------------------------------------------

with tab3:

    st.subheader("Key Business Insights")

    st.info("""
    • High-rated courses generate higher enrollments.

    • Mid-priced courses perform better than expensive courses.

    • Experienced instructors positively impact student demand.

    • Longer duration courses tend to generate more revenue.

    • Teacher ratings strongly influence course success.
    """)

    st.success("""
    Recommendation:
    EduPro should focus on high-quality instructors,
    optimize course pricing, and prioritize highly rated
    intermediate-level courses.
    """)

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.divider()

st.caption(
    "EduPro Predictive Analytics Dashboard | Developed using Python, Machine Learning & Streamlit"
)