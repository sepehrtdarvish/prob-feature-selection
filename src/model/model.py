from utils import load_dataset
from preproccess import Preproccess
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error

class RegressionModel():
    def __init__(self, df, top_cols_num=None, degree=1, model_type='xgboost'):
        self.df = df
        self.degree = degree
        self.top_cols_num = top_cols_num
        self.model_type = model_type
    
    def get_trainable_data(self):
        X = self.df.drop('SalePrice', axis=1)
        y = self.df['SalePrice']

        if self.top_cols_num:
            corr_matrix = self.df.corr()
            top_n_features = corr_matrix['SalePrice'].abs().sort_values(ascending=False).head(self.top_cols_num)
            top_n_features = top_n_features.iloc[1:].index.tolist()
            X = X[top_n_features]


        if self.degree > 1 and self.model_type in ['linear', 'ridge']:
            poly = PolynomialFeatures(degree=self.degree, include_bias=False)
            X_poly = poly.fit_transform(X)
            poly_columns = poly.get_feature_names_out(X.columns)
            X = pd.DataFrame(X_poly, columns=poly_columns)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test


    def train(self):
        X_train, X_test, y_train, y_test = self.get_trainable_data()

        if self.model_type == 'linear':
            model = LinearRegression()
        elif self.model_type == 'ridge':
            model = Ridge(alpha=1.0)
        elif self.model_type == 'random_forest':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif self.model_type == 'xgboost':
            model = XGBRegressor(
                n_estimators=1000,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.5,
                colsample_bytree=0.5,
                reg_alpha=0.02,
                n_jobs=-1)
        else:
            raise ValueError("Model type not supported")

        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        train_score = r2_score(y_train, y_pred_train)
        test_score = r2_score(y_test, y_pred_test)

        print(f"Model: {self.model_type}")
        print(f"Train R2 Score: {train_score:.4f}")
        print(f"Test R2 Score: {test_score:.4f}")
            
        return model
