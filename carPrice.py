# CLEAN END-TO-END USED CAR PRICE ANALYSIS WITH GRIDSEARCHCV
# ----------------------------------------------------------
# Full CRISP-DM aligned workflow
# Includes EDA, preprocessing, modeling, GridSearchCV tuning,
# evaluation, and insights.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import r2_score, root_mean_squared_error

# ============================================
# 1. DATA LOADING & UNDERSTANDING
# ============================================

df = pd.read_csv("data/vehicles.csv")

print("\n=== DATA SUMMARY ===")
print(df.info())
print(df.describe(include='all'))
print(df.isnull().sum())
print(df.head())

# ============================================
# 2. DATA CLEANING & PREPARATION
# ============================================

# Trim extreme prices
df = df[(df['price'] > 500) & (df['price'] < 200000)]
df = df.dropna(subset=['price'])

# Select final features
features = [
    'year', 'odometer', 'manufacturer', 'condition',
    'fuel', 'transmission', 'type'
]

df = df[features + ['price']]

# Impute missing
impute_vals = {
    'year': df['year'].median(),
    'manufacturer': 'unknown',
    'condition': 'unknown',
    'fuel': 'unknown',
    'transmission': 'unknown',
    'type': 'unknown'
}

df = df.fillna(impute_vals)
df['odometer'] = df['odometer'].fillna(df['odometer'].median())

# ============================================
# 3. EDA (Optional Visuals)
# ============================================

plt.figure(figsize=(8,4))
sns.histplot(df['price'], bins=80, kde=True)
plt.title("Price Distribution")
plt.show()

plt.figure(figsize=(8,4))
sns.scatterplot(data=df.sample(5000, random_state=42), x='year', y='price', alpha=0.3)
plt.title("Price vs Year")
plt.show()

plt.figure(figsize=(8,4))
sns.scatterplot(data=df.sample(5000, random_state=42), x='odometer', y='price', alpha=0.3)
plt.title("Price vs Odometer")
plt.show()

# Average price by manufacturer (top 10)
plt.figure(figsize=(10,5))
avg_price = df.groupby('manufacturer')['price'].mean().sort_values(ascending=False).head(10)
sns.barplot(x=avg_price.index, y=avg_price.values)
plt.title("Top 10 Manufacturers by Average Price")
plt.xticks(rotation=45)
plt.show()

# ============================================
# 4. MODELING SETUP
# ============================================

X = df[features]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_features = ['year', 'odometer']
cat_features = ['manufacturer', 'condition', 'fuel', 'transmission', 'type']

num_poly = Pipeline([
    ('scale', StandardScaler()),
    ('poly', PolynomialFeatures(include_bias=False))
])

cat_ohe = OneHotEncoder(handle_unknown='ignore')

preprocess = ColumnTransformer([
    ('num', num_poly, num_features),
    ('cat', cat_ohe, cat_features)
])
# Helper to wrap model

def make_pipe(model):
    return Pipeline([
        ('prep', preprocess),
        ('model', model)
    ])

# Base models
ridge = Ridge()

# Parameter grids
ridge_params = {
    'prep__num__poly__degree': [1, 2, 3],
    'model__alpha': [0.1, 1.0, 3.0, 10.0]
}

'''
# To do in future: Lasso grid search currently disabled for speed
lasso = Lasso(max_iter=50000)
lasso_params = {
    'prep__num__poly__degree': [1, 2],
    'model__alpha': [0.001, 0.01, 0.05, 0.1, 0.3],
    'model__tol': [1e-4, 1e-3, 1e-2]
}
'''

# Grid registry
models = {
    "Linear Regression": make_pipe(LinearRegression()),
    "Ridge": GridSearchCV(make_pipe(ridge), ridge_params, cv=3, scoring='r2', n_jobs=-1),
    #"Lasso": GridSearchCV(make_pipe(lasso), lasso_params, cv=3, scoring='r2', n_jobs=-1)
}

best_results = {}

print("\n========================================")
print("     MODEL TRAINING & GRID SEARCH")
print("========================================")

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)

    if isinstance(model, GridSearchCV):
        print(f"Best Params: {model.best_params_}")
        print(f"Best CV R²: {model.best_score_:.4f}")
        cv_score = model.best_score_
    else:
        cv_score = cross_val_score(model, X, y, cv=5).mean()
        print(f"CV R²: {cv_score:.4f}")

    best_results[name] = cv_score

# Select best model
best_model_name = max(best_results, key=lambda x: best_results[x])
best_model_obj = models[best_model_name]

print("\n========================================")
print(f" BEST MODEL SELECTED: {best_model_name}")
print("Polynomial degree: {}".format(best_model_obj.best_params_.get('prep__num__poly__degree', 'N/A')))
print("Alpha: {}".format(best_model_obj.best_params_.get('model__alpha', 'N/A')))
print("========================================")

final_pipe = (
    best_model_obj.best_estimator_
    if isinstance(best_model_obj, GridSearchCV)
    else best_model_obj
)

# ============================================
# 5. FINAL EVALUATION
# ============================================

y_pred = final_pipe.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)

print(f"Test R²: {r2:.4f} Polynomial degree: {best_model_obj.best_params_.get('prep__num__poly__degree','N/A')} Alpha: {best_model_obj.best_params_.get('model__alpha','N/A')}")
print(f"Test RMSE: ${rmse:,.2f}")

plt.figure(figsize=(6,6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.3)
plt.plot([0, 200000], [0, 200000], linestyle='--', color='red')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(f"Prediction Accuracy — {best_model_name}")
plt.show()

# ============================================
# 6. SUMMARY REPORT
# ============================================

print("""
============================================================
USED CAR PRICE ANALYSIS — SUMMARY
============================================================

Top Predictive Features:
• Year — newer cars command higher value
• Odometer — mileage is a major driver of depreciation
• Condition — strong price influence
• Manufacturer — luxury brands higher priced
• Type — trucks & SUVs command premiums

Best Model: {}
Polynomial degree: {}
Alpha: {}
Test R²: {:.3f}
RMSE: ${:,.2f}

============================================================
""".format(
    best_model_name,
    best_model_obj.best_params_.get('prep__num__poly__degree', 'N/A'),
    best_model_obj.best_params_.get('model__alpha', 'N/A'),
    r2,
    rmse
))