import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv('german_credit_data.csv')

# Show column names for debugging
print("Available columns in CSV:")
print(df.columns.tolist())

# Rename 'Risk' to 'default' and convert values to binary (example: 'good' -> 0, 'bad' -> 1)
df.rename(columns={'Risk': 'default'}, inplace=True)
df['default'] = df['default'].map({'good': 0, 'bad': 1})

# Drop rows with missing values
df = df.dropna()

# Select numeric and relevant features only (update as needed)
features = ['Age', 'Job', 'Credit amount', 'Duration']
X = df[features]
y = df['default']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Save model and scaler
joblib.dump(rf_model, 'rf_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model and scaler saved successfully.")
