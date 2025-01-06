#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# streamlit_app.py
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Set Streamlit page configuration
st.set_page_config(page_title="Fetal Health Classifier", layout="wide")

st.title("Fetal Health Classification with Machine Learning Models")

# File uploader for dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type="csv")

if uploaded_file:
    # Load the dataset
    df = pd.read_csv(uploaded_file)

    # Task 1: Inspect data
    st.subheader("Dataset Overview")
    st.write(df.head())
    st.write("Dataset Info:")
    buffer = df.info(buf=None)
    st.text(buffer)

    # Task 2: Handle duplicates
    duplicates = df.duplicated().sum()
    st.write(f"Total number of duplicate rows: {duplicates}")
    if duplicates > 0:
        df = df.drop_duplicates()
        st.write(f"Duplicate rows removed. Remaining duplicates: {df.duplicated().sum()}")

    # Task 3: Check for missing values
    null_values = df.isnull().sum()
    st.subheader("Missing Values Per Column")
    st.write(null_values)

    # Task 4: Analyze class distribution
    st.subheader("Class Distribution")
    class_distribution = df['fetal_health'].value_counts()
    st.bar_chart(class_distribution)

    # Task 5: Split data into features and target
    x = df.drop(columns=['fetal_health'])
    y = df['fetal_health']

    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    # Define a function for model evaluation
    def evaluate_model(model, x_test, y_test):
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy, y_pred

    # Train models
    logistic_model = LogisticRegression(solver='liblinear').fit(x_train, y_train)
    random_forest_model = RandomForestClassifier(
        criterion='gini', n_estimators=100, max_depth=4, random_state=33
    ).fit(x_train, y_train)
    decision_tree_model = DecisionTreeClassifier(criterion='gini', max_depth=2, random_state=33).fit(x_train, y_train)

    # Input interface for new patient data
    st.subheader("Predict Fetal Health for New Patient Data")
    input_data = {}
    for column in x.columns:
        value = st.number_input(f"Enter value for {column}", key=column)
        input_data[column] = value

    if st.button("Predict"):
        input_df = pd.DataFrame([input_data])
        st.write("Input Data:")
        st.write(input_df)

        # Make predictions using all models
        logistic_prediction = logistic_model.predict(input_df)[0]
        random_forest_prediction = random_forest_model.predict(input_df)[0]
        decision_tree_prediction = decision_tree_model.predict(input_df)[0]

        st.subheader("Prediction Results")
        st.write(f"**Logistic Regression Prediction:** {logistic_prediction}")
        st.write(f"**Random Forest Prediction:** {random_forest_prediction}")
        st.write(f"**Decision Tree Prediction:** {decision_tree_prediction}")

    # Task 9: Display confusion matrices
    st.subheader("Confusion Matrices")
    models = [
        ("Logistic Regression", logistic_model),
        ("Random Forest", random_forest_model),
        ("Decision Tree", decision_tree_model),
    ]
    for name, model in models:
        y_pred = model.predict(x_test)
        cm = confusion_matrix(y_test, y_pred)
        st.write(f"Confusion Matrix for {name}:")
        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        disp.plot(ax=ax)
        st.pyplot(fig)

    # Predictions Overview
    st.subheader("Predictions Overview")
    predictions_df = pd.DataFrame({
        'Actual': y_test.values,
        'Logistic Regression': logistic_model.predict(x_test),
        'Random Forest': random_forest_model.predict(x_test),
        'Decision Tree': decision_tree_model.predict(x_test)
    })
    st.write(predictions_df.head())

# Run the app using:
# streamlit run streamlit_app.py