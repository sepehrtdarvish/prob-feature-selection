import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error

class RegressionModel():
    """
    A wrapper for various regression models that handles training and evaluation.
    """
    def __init__(self, model_type='xgboost'):
        """
        Initializes the RegressionModel.

        Args:
            model_type (str): The type of model to use.
                              Options: 'linear', 'ridge', 'random_forest', 'xgboost'.
        """
        self.model_type = model_type
        self.model = self._get_model()

    def _get_model(self):
        """Internal method to instantiate the model based on model_type."""
        if self.model_type == 'linear':
            return LinearRegression()
        elif self.model_type == 'ridge':
            return Ridge(alpha=1.0)
        elif self.model_type == 'random_forest':
            return RandomForestRegressor(n_estimators=100, random_state=42)
        elif self.model_type == 'xgboost':
            return XGBRegressor(
                n_estimators=1000,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.5,
                colsample_bytree=0.5,
                reg_alpha=0.02,
                n_jobs=-1,
                random_state=42)
        else:
            raise ValueError(f"Model type '{self.model_type}' not supported")

    def train(self, X_train, X_test, y_train, y_test):
        """
        Trains the model and evaluates its performance.

        Args:
            X_train (pd.DataFrame or np.ndarray): Processed training features.
            X_test (pd.DataFrame or np.ndarray): Processed testing features.
            y_train (pd.Series or np.ndarray): Training target (log-transformed).
            y_test (pd.Series or np.ndarray): Testing target (log-transformed).
        """
        self.model.fit(X_train, y_train)

        # Make predictions on training and testing sets
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)

        # Evaluate performance on the log-transformed scale
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)

        # For interpretability, evaluate RMSE on the original dollar scale
        # by reversing the log-transformation.
        y_train_orig = np.expm1(y_train)
        y_test_orig = np.expm1(y_test)
        y_pred_train_orig = np.expm1(y_pred_train)
        y_pred_test_orig = np.expm1(y_pred_test)

        train_rmse_orig = np.sqrt(mean_squared_error(y_train_orig, y_pred_train_orig))
        test_rmse_orig = np.sqrt(mean_squared_error(y_test_orig, y_pred_test_orig))

        print(f"--- Model: {self.model_type} ---")
        print(f"Train R2: {train_r2:.4f} | Test R2: {test_r2:.4f}")
        print(f"Train MSE (log scale): {train_mse:.4f} | Test MSE (log scale): {test_mse:.4f}")
        print(f"Train RMSE (original scale): ${train_rmse_orig:,.2f}")
        print(f"Test RMSE (original scale): ${test_rmse_orig:,.2f}")
            
        return self.model