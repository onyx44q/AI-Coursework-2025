
#  STEP 1: Import Required Libraries


# For data handling
import pandas as pd
import numpy as np

# For visualization
import seaborn as sns
import matplotlib.pyplot as plt

# For modeling and preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score



#  STEP 2: Load the Diamonds Dataset


diamonds_df = sns.load_dataset('diamonds')

print("Original Data (first 5 rows):")
print(diamonds_df.head())


#  STEP 3: Data Cleaning


# Let's make sure our dataset is clean and ready for analysis.

# 1. Check for missing values
print(f"\nMissing values per column:\n{diamonds_df.isnull().sum()}")
# (Good news — there are no missing values in this dataset!)

# 2. Check for duplicate rows
print(f"\nNumber of duplicate rows: {diamonds_df.duplicated().sum()}")
diamonds_df = diamonds_df.drop_duplicates()
print(f"Dropped duplicates. New shape: {diamonds_df.shape}")

# 3. Look for "impossible" values — diamonds with zero dimensions
impossible_dims = (diamonds_df['x'] == 0) | (diamonds_df['y'] == 0) | (diamonds_df['z'] == 0)
print(f"\nFound {impossible_dims.sum()} diamonds with zero-dimensions.")

# Remove those invalid rows
diamonds_df = diamonds_df[~impossible_dims]
print(f"Removed impossible rows. Final shape: {diamonds_df.shape}")



# STEP 4: Exploratory Data Analysis (EDA)


# Let's visualize some key relationships in the data to understand patterns and trends.
sns.set_style("whitegrid")  # Use a clean, consistent plot style


# How does carat (weight) affect price?
plt.figure(figsize=(10, 6))
sns.scatterplot(x='carat', y='price', data=diamonds_df, alpha=0.1)
plt.title('Price vs. Carat')
plt.xlabel('Carat')
plt.ylabel('Price')
plt.show()

# Observation:
# There's a strong, positive, and somewhat exponential relationship:
# as carat weight increases, price rises dramatically.


# How does the quality of the 'cut' affect price?
plt.figure(figsize=(10, 6))
sns.boxplot(
    x='cut',
    y='price',
    data=diamonds_df,
    order=['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']  # Order by quality
)
plt.title('Price Distribution by Cut Quality')
plt.xlabel('Cut Quality')
plt.ylabel('Price')
plt.show()

# Observation:
# While 'Ideal' cuts have the highest median prices, 'Premium' and 'Very Good' cuts
# have some extremely high outliers. Once a cut is “good enough,”
# factors like carat weight seem to play a bigger role in pricing.


#  How do all numeric features relate to each other?
plt.figure(figsize=(10, 7))
corr_matrix = diamonds_df.corr(numeric_only=True)
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Numeric Features')
plt.show()

# Observation:
# Price is very strongly correlated with carat (0.92) and dimensions (x, y, z).
# Carat and the dimensions are also almost perfectly correlated with each other,
# which makes sense — heavier diamonds are bigger diamonds.
# Depth and table, on the other hand, show only weak correlations with price.



#  STEP 5: Prepare Data for Modeling


# We'll take a random sample to make training faster while keeping results consistent.
diamonds_model = diamonds_df.sample(n=12500, random_state=42)
print(f"\nCreated 'diamonds_model' sample with shape: {diamonds_model.shape}")


# Define the logical order of categories for ordinal encoding
cut_order = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
color_order = ['J', 'I', 'H', 'G', 'F', 'E', 'D']
clarity_order = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']

# Make a copy to preserve the original sample
df_model1 = diamonds_model.copy()

# Encode the categorical features numerically based on their quality order
encoder = OrdinalEncoder(categories=[cut_order, color_order, clarity_order], dtype=int)
df_model1[['cut', 'color', 'clarity']] = encoder.fit_transform(df_model1[['cut', 'color', 'clarity']])

print("\nData after Ordinal Encoding (first 5 rows):")
print(df_model1.head())



#  STEP 6: Train-Test Split & Scaling


# Separate features (X) from the target (y)
X = df_model1.drop('price', axis=1)
y = df_model1['price']

# Split data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize all features (important since all are numeric now)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



#  STEP 7: Model 1 - Full Linear Regression


model_lr = LinearRegression()
model_lr.fit(X_train_scaled, y_train)

# Evaluate model performance
y_pred_lr = model_lr.predict(X_test_scaled)
r2_lr = r2_score(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))

print("\n--- Model 1: Full Linear Regression Results ---")
print(f"R-squared (R²): {r2_lr:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_lr:.2f}")

# Store model results for comparison
model_scores = {
    'Full Linear Regression': {'R²': r2_lr, 'RMSE': rmse_lr}
}



#  STEP 8: Model 2 - PCA Regression (Dimensionality Reduction)


# Select only continuous variables for PCA
continuous_features = ['carat', 'depth', 'table', 'x', 'y', 'z']
y_pca = diamonds_model['price']
X_continuous = diamonds_model[continuous_features]

# PCA requires scaled data
scaler_pca = StandardScaler()
X_continuous_scaled = scaler_pca.fit_transform(X_continuous)

# Reduce features to 2 principal components
pca = PCA(n_components=2)
X_pca_components = pca.fit_transform(X_continuous_scaled)

print(f"\nExplained variance by 2 PCA components: {pca.explained_variance_ratio_.sum():.4f}")
# This shows how much information the 2 PCA components preserve.

# Split PCA data for training and testing
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(
    X_pca_components, y_pca, test_size=0.2, random_state=42
)

# Train and evaluate the PCA-based model
model_pca_lr = LinearRegression()
model_pca_lr.fit(X_train_pca, y_train_pca)

y_pred_pca = model_pca_lr.predict(X_test_pca)
r2_pca = r2_score(y_test_pca, y_pred_pca)
rmse_pca = np.sqrt(mean_squared_error(y_test_pca, y_pred_pca))

print("\n--- Model 2: PCA (2 Components) Linear Regression Results ---")
print(f"R-squared (R²): {r2_pca:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_pca:.2f}")

model_scores['PCA Regression'] = {'R²': r2_pca, 'RMSE': rmse_pca}


#  STEP 9: Model 3 - Lasso Regression (L1 Regularization)


# Lasso helps in feature selection by shrinking less important coefficients to zero
model_lasso = Lasso(alpha=1.0, random_state=42)
model_lasso.fit(X_train_scaled, y_train)

# Evaluate performance
y_pred_lasso = model_lasso.predict(X_test_scaled)
r2_lasso = r2_score(y_test, y_pred_lasso)
rmse_lasso = np.sqrt(mean_squared_error(y_test, y_pred_lasso))

print("\n--- Model 3: Lasso Regression Results ---")
print(f"R-squared (R²): {r2_lasso:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_lasso:.2f}")

model_scores['Lasso Regression'] = {'R²': r2_lasso, 'RMSE': rmse_lasso}



#  STEP 10: Model 4 - Ridge Regression (L2 Regularization)


# Ridge regression penalizes large coefficients to reduce overfitting
model_ridge = Ridge(alpha=1.0, random_state=42)
model_ridge.fit(X_train_scaled, y_train)

# Evaluate performance
y_pred_ridge = model_ridge.predict(X_test_scaled)
r2_ridge = r2_score(y_test, y_pred_ridge)
rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))

print("\n--- Model 4: Ridge Regression Results ---")
print(f"R-squared (R²): {r2_ridge:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_ridge:.2f}")

model_scores['Ridge Regression'] = {'R²': r2_ridge, 'RMSE': rmse_ridge}



# STEP 11: Final Model Comparison


# Combine all results into a neat comparison table
scores_df = pd.DataFrame(model_scores).T
scores_df = scores_df.sort_values(by='R²', ascending=False)

print("\nFinal Model Comparison")
print(scores_df)
