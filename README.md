# Fetal Health Classification using Python

This project focuses on leveraging machine learning techniques to classify fetal health into three categories: Normal, Suspect, and Pathological, based on data obtained from Cardiotocogram (CTG) exams. The dataset consists of 2,126 records, each containing 21 features that describe various measurements related to fetal health.

Objectives:
1. Build and evaluate machine learning models to accurately predict fetal health status.
2. Analyze and address misclassifications to improve model reliability.
3. Provide insights that can support healthcare professionals in early identification of potential health issues.

Approach:
1. Data Preparation:
    • Data was preprocessed to handle duplicates, null values, and inconsistencies.
    • The dataset was split into training (70%) and testing (30%) sets for model evaluation.

2. Models Implemented:
    • Logistic Regression: A baseline linear classifier to distinguish between the three classes.
    • Random Forest Classifier: A robust ensemble model using 100 trees, with a depth constraint to prevent overfitting.
    • Decision Tree Classifier: A simplified decision tree with limited depth to evaluate performance against more complex models.

3. Evaluation Metrics:
    • Model performance was evaluated using accuracy, confusion matrix, and classification reports to analyze precision, recall, and F1-score for each class.
    • A detailed review of misclassified cases was performed to understand common prediction errors.

4. Handling Class Imbalance:
    • Class distribution was analyzed, revealing potential bias towards one class.
    • To address this, the Random Forest model was updated with a class_weight='balanced' parameter, ensuring fair treatment of underrepresented classes.

5. Feature Importance:
    • Feature importance analysis was conducted to identify the most critical factors influencing predictions.
    • Insights were used to refine the models and guide potential feature engineering.


