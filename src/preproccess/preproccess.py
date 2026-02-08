import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class Preprocessor:
    """
    Creates a preprocessing pipeline for tabular data.

    The pipeline handles imputation, scaling for numeric features, and
    one-hot encoding for categorical features. This ensures that steps are
    fit only on the training data, preventing data leakage.
    """

    def create_pipeline(self, numeric_features, categorical_features):
        """
        Builds and returns a scikit-learn pipeline for preprocessing.

        Args:
            numeric_features (list): List of names of numeric columns.
            categorical_features (list): List of names of categorical columns.

        Returns:
            sklearn.pipeline.Pipeline: The configured preprocessing pipeline.
        """
        # Define the processing steps for numeric features.
        # 1. Impute missing values with the median of the column.
        # 2. Scale features to have zero mean and unit variance.
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # Define the processing steps for categorical features.
        # 1. Impute missing values with a constant string 'missing'.
        # 2. One-hot encode the categories. `handle_unknown='ignore'` prevents
        #    errors if a category seen in testing was not in training.
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False))
        ])

        # Create a ColumnTransformer to apply the correct transformations
        # to the correct columns.
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='passthrough'  # Keep other columns if they exist
        )

        # The final pipeline consists of the preprocessor. This can be
        # chained with a model in a larger pipeline if desired.
        return Pipeline(steps=[('preprocessor', preprocessor)])