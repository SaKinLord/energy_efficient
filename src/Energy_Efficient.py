# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 14:00:44 2025
@author: merto
"""

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, learning_curve
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.compose import ColumnTransformer

# ----------------------------- Data Loading and Preprocessing -----------------------------

# Load the dataset
energy = pd.read_csv('ENB2012_data.csv')

# Print dataset information and check for missing values
print(energy.info())
print(energy.describe())
print(energy.isnull().sum())

# ----------------------------- Data Visualization -----------------------------

# Calculate the correlation matrix
corr = energy.corr()

# Plot heatmap to visualize correlations
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", annot_kws={"size": 10})
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title("Correlation Matrix", fontsize=16)
plt.show()

# Scatter plots for features vs Y1 (Heating Load)
features = ['X1', 'X2', 'X3', 'X4', 'X5']
plt.figure(figsize=(20, 4))
for i, feat in enumerate(features):
    plt.subplot(1, 5, i + 1)
    plt.scatter(energy[feat], energy['Y1'], alpha=0.7, edgecolor='k')
    plt.xlabel(feat)
    plt.ylabel('Y1')
    plt.title(f'{feat} vs Y1')
plt.tight_layout()
plt.show()

# Scatter plots for features vs Y2 (Cooling Load)
plt.figure(figsize=(20, 4))
for i, feat in enumerate(features):
    plt.subplot(1, 5, i + 1)
    plt.scatter(energy[feat], energy['Y2'], alpha=0.7, edgecolor='k')
    plt.xlabel(feat)
    plt.ylabel('Y2')
    plt.title(f'{feat} vs Y2')
plt.tight_layout()
plt.show()

# Box plot for selected variables
columns = ['X1', 'X2', 'X3', 'X4', 'X5', 'Y1', 'Y2']
plt.figure(figsize=(12, 6))
energy[columns].boxplot(patch_artist=True,
                        boxprops=dict(facecolor='lightgreen'),
                        medianprops=dict(color='red'))
plt.title("Box Plot for Selected Variables")
plt.ylabel("Values")
plt.show()

# Pairplot for selected variables
sns.pairplot(energy[columns], diag_kind='kde', corner=True)
plt.show()

# Check the correlation between the target variables (Y1 and Y2)
single_or_multi = energy[['Y1', 'Y2']].corr()

# ----------------------------- Data Preparation -----------------------------

# For XGBoost, select specific features: columns at indices 0, 2, and 6
X_xgb = energy.iloc[:, [0, 2, 6]].values  
y_xgb = energy.iloc[:, 8:].values  # Target variables (Y1 and Y2)

# For KNN, use all features except the last two (Y1 and Y2)
X_knn = energy.iloc[:, :-2].values  
y_knn = energy.iloc[:, 8:].values  # Target variables

# Split the data into training and testing sets (20% test size)
X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(
    X_xgb, y_xgb, test_size=0.2, random_state=42
)

X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(
    X_knn, y_knn, test_size=0.2, random_state=42
)

# Standardize features for KNN (excluding the first column)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), list(range(1, X_knn.shape[1])))  # Standardize columns from index 1 onwards
    ],
    remainder='passthrough'  # Leave the first column as is
)

X_train_knn = preprocessor.fit_transform(X_train_knn)
X_test_knn = preprocessor.transform(X_test_knn)

# ----------------------------- XGBoost Model -----------------------------

# Define the parameter grid for XGBoost
param_dist_xgb = {
    'estimator__n_estimators': [100, 200, 300],
    'estimator__learning_rate': [0.01, 0.05, 0.1],
    'estimator__max_depth': [3, 4, 5],
    'estimator__subsample': [0.8, 0.9, 1.0],
    'estimator__colsample_bytree': [0.8, 0.9, 1.0],
    'estimator__gamma': [0, 0.1, 0.2],
    'estimator__reg_lambda': [0, 1.0, 10.0],
    'estimator__reg_alpha': [0, 1.0, 10.0]
}

# Wrap XGBRegressor with MultiOutputRegressor to handle multiple outputs
multi_xgb = MultiOutputRegressor(XGBRegressor(random_state=42))

# Use RandomizedSearchCV for hyperparameter optimization
random_search_xgb = RandomizedSearchCV(
    estimator=multi_xgb,
    param_distributions=param_dist_xgb,
    n_iter=100,  # Number of iterations (if total combinations are fewer, all combinations will be tried)
    scoring='r2',
    cv=5,
    random_state=42,
    n_jobs=-1
)

# Train the XGBoost model
random_search_xgb.fit(X_train_xgb, y_train_xgb)  # No need for ravel() due to multi-output

# Print the best parameters and best cross-validation R² score for XGBoost
print("Best XGBoost parameters:", random_search_xgb.best_params_)
print("Best XGBoost R² score:", random_search_xgb.best_score_)

# Make predictions with the best XGBoost model
best_xgb = random_search_xgb.best_estimator_
predictions_best_xgb = best_xgb.predict(X_test_xgb)

# Evaluate performance metrics for XGBoost
mae_best_xgb = mean_absolute_error(y_test_xgb, predictions_best_xgb)
mse_best_xgb = mean_squared_error(y_test_xgb, predictions_best_xgb)
rmse_best_xgb = np.sqrt(mse_best_xgb)
r2_best_xgb = r2_score(y_test_xgb, predictions_best_xgb)

# Print performance metrics for XGBoost
print(f"📌 **Optimized XGBoost Results:**")
print(f"MAE: {mae_best_xgb:.4f}")
print(f"MSE: {mse_best_xgb:.4f}")
print(f"RMSE: {rmse_best_xgb:.4f}")
print(f"R² Score: {r2_best_xgb:.4f}")

# ----------------------------- KNN Model -----------------------------

# Define the parameter grid for KNN
param_grid_knn = {
    'estimator__n_neighbors': [3, 5, 7, 9],  # Number of neighbors
    'estimator__weights': ['uniform', 'distance'],  # Weighting method
    'estimator__metric': ['euclidean', 'manhattan']  # Distance metric
}

# Wrap KNeighborsRegressor with MultiOutputRegressor for multi-output regression
multi_knn = MultiOutputRegressor(KNeighborsRegressor())

# Use GridSearchCV for hyperparameter optimization for KNN
grid_search_knn = GridSearchCV(
    estimator=multi_knn,
    param_grid=param_grid_knn,
    scoring='r2',
    cv=5,
    n_jobs=-1
)

# Train the KNN model
grid_search_knn.fit(X_train_knn, y_train_knn)

# Print the best parameters and best cross-validation R² score for KNN
print("Best KNN parameters:", grid_search_knn.best_params_)
print("Best KNN R² score:", grid_search_knn.best_score_)

# Make predictions with the best KNN model
best_knn = grid_search_knn.best_estimator_
predictions_best_knn = best_knn.predict(X_test_knn)

# Evaluate performance metrics for KNN
mae_best_knn = mean_absolute_error(y_test_knn, predictions_best_knn)
mse_best_knn = mean_squared_error(y_test_knn, predictions_best_knn)
rmse_best_knn = np.sqrt(mse_best_knn)
r2_best_knn = r2_score(y_test_knn, predictions_best_knn)

# Print performance metrics for KNN
print(f"📌 **Optimized KNN Results:**")
print(f"MAE: {mae_best_knn:.4f}")
print(f"MSE: {mse_best_knn:.4f}")
print(f"RMSE: {rmse_best_knn:.4f}")
print(f"R² Score: {r2_best_knn:.4f}")

# ----------------------------- Comparison Table -----------------------------

print("\n📌 **Comparison Table:**")
print(f"{'Metric':<10} | {'XGBoost':<10} | {'KNN':<10}")
print("-" * 35)
print(f"{'MAE':<10} | {mae_best_xgb:<10.4f} | {mae_best_knn:<10.4f}")
print(f"{'MSE':<10} | {mse_best_xgb:<10.4f} | {mse_best_knn:<10.4f}")
print(f"{'RMSE':<10} | {rmse_best_xgb:<10.4f} | {rmse_best_knn:<10.4f}")
print(f"{'R² Score':<10} | {r2_best_xgb:<10.4f} | {r2_best_knn:<10.4f}")

# ----------------------------- Observed vs. Predicted Plot -----------------------------

# Plot Observed vs. Predicted values for XGBoost model
plt.figure(figsize=(8, 6))
plt.scatter(y_test_xgb[:, 0], predictions_best_xgb[:, 0], color='blue', label='Heating Load (Y1)', alpha=0.7)
plt.scatter(y_test_xgb[:, 1], predictions_best_xgb[:, 1], color='green', label='Cooling Load (Y2)', alpha=0.7)
plt.plot([y_test_xgb.min(), y_test_xgb.max()], [y_test_xgb.min(), y_test_xgb.max()], color='red', linestyle='--', label='Perfect Fit')
plt.xlabel("Observed Values")
plt.ylabel("Predicted Values")
plt.title("Observed vs. Predicted (XGBoost)")
plt.legend()
plt.show()

# ----------------------------- Learning Curve -----------------------------

def plot_learning_curve(estimator, X, y, cv=5, scoring='r2'):
    """
    Plots the learning curve to observe if the model is overfitting.
    
    Parameters:
        estimator: The model/estimator to evaluate.
        X: Feature data.
        y: Target data.
        cv: Number of cross-validation folds.
        scoring: Scoring metric (default is 'r2').
    """
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring,
        train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_scores_mean, label="Training Score", marker='o')
    plt.plot(train_sizes, val_scores_mean, label="Validation Score", marker='o')
    plt.xlabel("Training Data Size")
    plt.ylabel("R² Score")
    plt.title("Learning Curve for XGBoost")
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot the learning curve for the best XGBoost model using the full X_xgb and y_xgb data
plot_learning_curve(best_xgb, X_xgb, y_xgb)
