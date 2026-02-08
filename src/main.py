import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

from utils.load_dataset import load_dataset
from preproccess.preproccess import Preprocessor
from model.model import RegressionModel

def main():
    # --- Configuration ---
    TARGET_COL = 'SalePrice'
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    TOP_N_FEATURES = 20  # Set to None to use all features
    POLYNOMIAL_DEGREE = 1
    MODEL_TYPE = 'xgboost' # Options: 'linear', 'ridge', 'random_forest', 'xgboost'

    # 1. Load Data
    df = load_dataset()
    
    # 2. Separate target variable and apply log transform to it
    X = df.drop(TARGET_COL, axis=1)
    y = np.log1p(df[TARGET_COL]) # Use log transform for skewed target

    # 3. Split Data *before* any fitting or selection to prevent data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    # 4. Feature Engineering (Example: log transform skewed features)
    # This transformation is safe to do before the pipeline
    if 'Lot Area' in X_train.columns:
        # Use .copy() to avoid SettingWithCopyWarning
        X_train = X_train.copy()
        X_test = X_test.copy()
        X_train['Lot Area'] = np.log1p(X_train['Lot Area'])
        X_test['Lot Area'] = np.log1p(X_test['Lot Area'])

    # 5. Feature Selection (applied only on training data)
    if TOP_N_FEATURES:
        print(f"\nPerforming feature selection for top {TOP_N_FEATURES} features...")
        # Temporarily combine X_train and y_train to calculate correlations
        temp_train_df = pd.concat([X_train, y_train.rename(TARGET_COL)], axis=1)
        
        # Calculate correlations with the target (numeric only)
        corr_matrix = temp_train_df.corr(numeric_only=True)
        # Select top N features, excluding the target variable itself (at index 0)
        top_cols = corr_matrix[TARGET_COL].abs().sort_values(ascending=False).index[1:TOP_N_FEATURES+1].tolist()
        
        print(f"Selected features: {top_cols}")
        
        # Filter datasets to only include these top features
        X_train = X_train[top_cols]
        X_test = X_test[top_cols]

    # 6. Create and Apply Preprocessing Pipeline
    # Identify column types from the (potentially filtered) training set
    numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_train.select_dtypes(exclude=np.number).columns.tolist()

    preprocessor_builder = Preprocessor()
    preprocessing_pipeline = preprocessor_builder.create_pipeline(numeric_features, categorical_features)

    # Fit the pipeline on the training data and transform both sets
    X_train_processed = preprocessing_pipeline.fit_transform(X_train)
    X_test_processed = preprocessing_pipeline.transform(X_test)
    
    # 7. Add Polynomial Features (Optional, after preprocessing)
    if POLYNOMIAL_DEGREE > 1 and MODEL_TYPE in ['linear', 'ridge']:
        poly = PolynomialFeatures(degree=POLYNOMIAL_DEGREE, include_bias=False)
        X_train_processed = poly.fit_transform(X_train_processed)
        X_test_processed = poly.transform(X_test_processed)

    # 8. Train and Evaluate the Model
    model = RegressionModel(model_type=MODEL_TYPE)
    model.train(X_train_processed, X_test_processed, y_train, y_test)


if __name__ == '__main__':
    main()