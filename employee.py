# Cell 0: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os # <-- IMPORT ADDED FOR ROBUST PATH HANDLING

# Preprocessing and Model Selection
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC  # Support Vector Classifier

# Evaluation Metrics
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

print("Libraries imported successfully.")

# --------------------------------------------------------------------
# Cell 1: Load the Dataset (Q1) - CORRECTED
# --------------------------------------------------------------------

# Attempt to build a robust file path (assuming the CSV is next to the .py script)
script_dir = os.path.dirname(os.path.realpath(__file__))
file_path = os.path.join(script_dir, "EmployeeAttrition.csv")

# Initialize df to None outside the try block
df = None

try:
    # Read using the full path
    df = pd.read_csv(file_path)
    print("Employee Attrition dataset loaded successfully!")
    print(df.head())
except FileNotFoundError:
    print("="*50)
    print(f"ERROR: 'employee_attrition.csv' not found at path: {file_path}")
    print("Please ensure the CSV file is in the same folder as your Python script.")
    print("="*50)

# --------------------------------------------------------------------
# Check if df was loaded before proceeding (Fixes the NameError)
# --------------------------------------------------------------------
if df is not None:
    # Cell 2: Data Cleaning & Exploratory Data Analysis (Q2) - NOW CORRECTLY UN-INDENTED

    # --- 1. Data Cleaning ---
    print("--- 1. Data Cleaning ---")
    print(f"Initial shape of data: {df.shape}")

    # Check for missing values
    print(f"\nTotal missing values: {df.isnull().sum().sum()}")

    # Check for duplicates
    print(f"Total duplicate rows: {df.duplicated().sum()}")

    # Drop non-informative columns
    cols_to_drop = ['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber']
    df_cleaned = df.drop(columns=cols_to_drop)

    print(f"Dropped non-informative columns.")

    # --- 2. Preprocessing for Modeling ---
    # Convert the target variable 'Attrition' from Yes/No to 1/0
    le = LabelEncoder()
    df_cleaned['Attrition'] = le.fit_transform(df_cleaned['Attrition'])

    # Create a fully numeric dataframe for the correlation matrix
    df_processed = pd.get_dummies(df_cleaned, drop_first=True)
    print("Converted 'Attrition' and other text columns to numeric.")

    # --- 3. Exploratory Data Analysis (4 Insights) ---
    print("\n--- 2. Exploratory Data Analysis ---")

    # Insight 1: Target Variable Imbalance
    plt.figure(figsize=(7, 5))
    # Note: Use original df columns if text labels are needed for charts
    sns.countplot(x=df['Attrition'])
    plt.title('Insight 1: Distribution of Employee Attrition')
    plt.show()
    print("Observation 1: The dataset is highly imbalanced. Far more employees stayed ('No') than left ('Yes').")

    # Insight 2: Attrition by Monthly Income
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Attrition', y='MonthlyIncome', data=df)
    plt.title('Insight 2: Monthly Income vs. Attrition')
    plt.show()
    print("Observation 2: Employees who left ('Yes') have a noticeably lower median monthly income.")

    # Insight 3: Attrition by OverTime
    plt.figure(figsize=(8, 5))
    sns.countplot(x='OverTime', hue='Attrition', data=df)
    plt.title('Insight 3: Attrition based on Working OverTime')
    plt.show()
    print("Observation 3: Employees who work overtime are significantly more likely to leave the company.")

    # Insight 4: Correlation Heatmap (using the processed numeric data)
    plt.figure(figsize=(12, 10))
    corr_matrix = df_processed.corr(numeric_only=True)
    sns.heatmap(corr_matrix.corr(numeric_only=True), annot=False, cmap='coolwarm', fmt='.2f')
    plt.title('Insight 4: Correlation Matrix of All Features')
    plt.show()
    print("Observation 4: The heatmap shows 'OverTime_Yes' has a positive correlation with Attrition, while 'MonthlyIncome', 'TotalWorkingYears', and 'Age' have negative correlations.")

    # Cell 3: k-Nearest Neighbours (k-NN) Model (Q3)

    # --- 1. Define Features (X) and Target (y) ---
    features = ['MonthlyIncome', 'TotalWorkingYears']
    target = 'Attrition'

    X = df_processed[features]
    y = df_processed[target]

    # --- 2. Train-Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # --- 3. Feature Scaling (CRITICAL for k-NN) ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Data split and scaled for k-NN.")

    # --- 4. Verify Optimal k (Elbow Method) ---
    error_rate = []
    for i in range(1, 31):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train_scaled, y_train)
        pred_i = knn.predict(X_test_scaled)
        error_rate.append(np.mean(pred_i != y_test))

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 31), error_rate, color='blue', linestyle='dashed', marker='o',
             markerfacecolor='red', markersize=10)
    plt.title('Error Rate vs. K Value')
    plt.xlabel('K')
    plt.ylabel('Error Rate')
    plt.show()

    # --- 5. Create Final k-NN Model ---
    optimal_k = 11 # Use the value from your plot
    knn_model = KNeighborsClassifier(n_neighbors=optimal_k)
    knn_model.fit(X_train_scaled, y_train)

    print(f"k-NN model created and trained with optimal k = {optimal_k}")

    # Cell 4: Eager Learning Classifier (SVM) (Q4)

    # --- 1. Create the Base SVM Model ---
    svm_model = SVC(random_state=42)

    # --- 2. Train the Model ---
    svm_model.fit(X_train_scaled, y_train)

    print("Base SVM model (Eager Learner) created and trained.")


    # Cell 5: Tuned SVM Model (Q5)

    print("Starting SVM hyperparameter tuning (this may take a minute)...")

    # --- 1. Define Parameter Grid ---
    param_grid = {
        'C': [0.1, 1, 10],            # Regularization parameter
        'kernel': ['linear', 'rbf'],  # Kernel type
        'gamma': ['scale', 'auto']    # Kernel coefficient for 'rbf'
    }

    # --- 2. Set up GridSearchCV ---
    grid_search = GridSearchCV(estimator=SVC(random_state=42),
                               param_grid=param_grid,
                               cv=5,
                               scoring='accuracy',
                               n_jobs=-1)

    # --- 3. Run the Search ---
    grid_search.fit(X_train_scaled, y_train)

    # --- 4. Get the Best Model ---
    print("\n--- Tuning Complete ---")
    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best accuracy during tuning: {grid_search.best_score_:.4f}")

    tuned_svm_model = grid_search.best_estimator_

    print("Tuned SVM model created.")

    # Cell 6: Evaluate Performances (Q6)

    print("--- 1. Evaluating k-NN Model (Q3) ---")
    y_pred_knn = knn_model.predict(X_test_scaled)
    print("Classification Report (k-NN):")
    print(classification_report(y_test, y_pred_knn))
    cm_knn = confusion_matrix(y_test, y_pred_knn)
    disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_knn, display_labels=['No', 'Yes'])
    disp_knn.plot(cmap='Blues')
    plt.title(f'k-NN Confusion Matrix (k={optimal_k})')
    plt.show()


    print("\n--- 2. Evaluating Base SVM Model (Q4) ---")
    y_pred_svm = svm_model.predict(X_test_scaled)
    print("Classification Report (Base SVM):")
    print(classification_report(y_test, y_pred_svm))
    cm_svm = confusion_matrix(y_test, y_pred_svm)
    disp_svm = ConfusionMatrixDisplay(confusion_matrix=cm_svm, display_labels=['No', 'Yes'])
    disp_svm.plot(cmap='Blues')
    plt.title('Base SVM Confusion Matrix')
    plt.show()


    print("\n--- 3. Evaluating Tuned SVM Model (Q5) ---")
    y_pred_tuned_svm = tuned_svm_model.predict(X_test_scaled)
    print("Classification Report (Tuned SVM):")
    print(classification_report(y_test, y_pred_tuned_svm))
    cm_tuned_svm = confusion_matrix(y_test, y_pred_tuned_svm)
    disp_tuned_svm = ConfusionMatrixDisplay(confusion_matrix=cm_tuned_svm, display_labels=['No', 'Yes'])
    disp_tuned_svm.plot(cmap='Blues')
    plt.title('Tuned SVM Confusion Matrix')
    plt.show()

else:
    print("\nSkipping model execution because the data file could not be loaded.")


