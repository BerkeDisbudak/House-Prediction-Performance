import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

# Load the data
X_all = pd.read_csv('./train.csv', index_col='Id')
X_test_all = pd.read_csv('./test.csv', index_col='Id')

# Separate the target and independent variables
X_all.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X_all.SalePrice
X_all.drop(['SalePrice'], axis=1, inplace=True)

# Only select numeric variables
X = X_all.select_dtypes(exclude=['object'])
X_test = X_test_all.select_dtypes(exclude=['object'])

# Split into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42)

# Define the pipeline
pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', MinMaxScaler()),
    ('model', XGBRegressor(n_estimators=1000, learning_rate=0.02, max_depth=5, random_state=42))
])

# Evaluate with cross-validation
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
mean_cv_score = -1 * np.mean(cv_scores)
print(f"Mean Cross-Validated MAE: {mean_cv_score:.2f}")

# Train the final model with the pipeline
pipeline.fit(X_train, y_train)
pred_valid = pipeline.predict(X_valid)

# MAE on the validation set
valid_mae = mean_absolute_error(y_valid, pred_valid)
print(f"Validation MAE: {valid_mae:.2f}")

# Predictions on the test set
pipeline.fit(X, y)  # Train on the entire dataset
pred_test_results = pipeline.predict(X_test)

# Histogram of the predictions

plt.figure(figsize=(10, 6))
sns.histplot(pred_test_results, bins=50, color='green', kde=True)
plt.title("Distribution of Predicted Sale Prices", fontsize=16)
plt.xlabel("Predicted Sale Price", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()