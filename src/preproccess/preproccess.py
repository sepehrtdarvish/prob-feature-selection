import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class Preproccess():
    def __init__(self, df):
        self.df = df


    def get_mostly_null_cols(self):
        return self.df.columns[self.df.isnull().sum() > (1/10) * len(self.df)].tolist()


    def get_low_null_cols(self):
        return self.df.columns[self.df.isnull().sum() == 1].tolist()
    

    def drop_null_cols(self):
        mostly_null_cols = self.get_mostly_null_cols()
        self.df = self.df.drop(mostly_null_cols, axis=1)


    def drop_null_rows(self):
        low_null_cols = self.get_low_null_cols()
        rows_with_nulls = self.df[self.df[low_null_cols].isnull().any(axis=1)]
        rows_index = rows_with_nulls.index
        self.df = self.df.drop(rows_index, axis=0)


    def fill_static_cols(self):
        self.df['Mas Vnr Area'] = self.df['Mas Vnr Area'].fillna(0)
        self.df['Bsmt Full Bath'] = self.df['Bsmt Full Bath'].fillna(0)
        self.df['Bsmt Half Bath'] = self.df['Bsmt Half Bath'].fillna(0)
        self.df['Garage Yr Blt'] = self.df['Garage Yr Blt'].fillna(0)

        cols_to_none = [
        'Bsmt Qual', 'Bsmt Cond', 'Bsmt Exposure',
        'BsmtFin Type 1', 'BsmtFin Type 2',
        'Garage Type', 'Garage Finish', 'Garage Qual', 'Garage Cond'
        ]

        self.df[cols_to_none] = self.df[cols_to_none].fillna('None')


    def conver_to_int(self):
        self.df = pd.get_dummies(self.df, drop_first=True)
        self.df = self.df.replace({True: 1, False: 0})


    def log_price(self):
        self.df['SalePrice'] = np.log1p(self.df['SalePrice'])

    def log_area(self):
        self.df['Lot Area'] = np.log1p(self.df['Lot Area'])
    
    def scale_features(self):
        scaler = StandardScaler()
        target = self.df['SalePrice']
        features = self.df.drop(columns=['SalePrice'])

        features_scaled = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)

        self.df = pd.concat([features_scaled, target.reset_index(drop=True)], axis=1)


    def remove_outliers(self, target_col='SalePrice', threshold=3):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        Q1 = self.df[numeric_cols].quantile(0.25)
        Q3 = self.df[numeric_cols].quantile(0.75)
        IQR = Q3 - Q1

        condition = ~((self.df[numeric_cols] < (Q1 - threshold * IQR)) | 
                      (self.df[numeric_cols] > (Q3 + threshold * IQR))).any(axis=1)
        
        self.df = self.df[condition]


    def run_all(self):
        self.drop_null_cols()
        self.drop_null_rows()
        self.fill_static_cols()

        self.remove_outliers()

        self.conver_to_int()
        self.log_price()
        self.log_area()
        #self.scale_features()

        return self.df