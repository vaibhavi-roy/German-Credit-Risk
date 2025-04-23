# German Credit Risk

This project centers on predicting credit risk using machine learning techniques. The main goal is to determine whether a loan applicant is likely to default, based on their profile. The workflow includes data analysis, preprocessing, feature engineering, model building, and evaluation.

## Libraries and DataSet

The following Python libraries are used in this project:

- `Pandas`
- `NumPy`
- `Seaborn`
- `Matplotlib`
- `Plotly`

The dataset used is **"german_credit_data.csv"**.

## Data Analysis

### Gender-Based Insights
- **69%** of the applicants are male, and **31%** are female.
- Female applicants tend to apply for loans at a younger age.
- The most common loan purposes are automobile purchases and consumer electronics (radio/TV).

### Risk Profiling
- Credit amount and loan duration have a significant impact on credit risk, with higher amounts and longer durations being associated with higher default risks.
- Applicants with richer housing types are less likely to default.
- Applicants in the age groups **20-30** and **60+** have a slightly higher chance of default.

## Data Preprocessing and Feature Engineering

- **Missing Value Handling**: Missing values in the "Saving accounts" and "Checking account" fields are filled with the value "none".
- **Feature Mapping**: "Job" feature values are mapped to more descriptive labels.
- **Age Group Creation**: An "Age group" feature is created by categorizing applicants into different age ranges.

## Outlier Detection

- Outliers in numeric features such as "Age", "Credit amount", and "Duration" are identified and handled.

## Feature Transformation

- **One-Hot Encoding**: Categorical variables are encoded using one-hot encoding.
- **Scaling**: Numeric features are standardized using robust scaling techniques to improve model performance.

## Machine Learning Models

Several machine learning models are tested for predicting credit risk, including:

- **Logistic Regression**
- **Decision Tree**
- **Naive Bayes**
- **K-Nearest Neighbors**
- **Random Forest**
- **Support Vector Machines (SVM)**
- **XGBoost**

Model performance is evaluated using cross-validation and **recall** as the primary scoring metric. The best-performing models are **Naive Bayes**, **Decision Tree**, and **XGBoost**.

## Hyperparameter Tuning

The top-performing models are fine-tuned using grid search techniques to optimize their performance. Metrics such as precision, recall, and ROC curves are used for model evaluation.

## Project Files

- `german_credit_data.csv`: The dataset used for training and testing the models.
- `app.py`: Streamlit app that provides an interactive interface for users to input credit application data and get predictions.
- `rf_model.pkl`: The trained model that predicts credit risk.
- `german_credit_risk.ipynb`: Jupyter Notebook containing detailed steps of the analysis, model building, and evaluation.
