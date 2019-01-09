import datetime
import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import DataConversionWarning

class Model():
    def __init__(self, models, coef):
        self.models = models
        self.coef = coef

    def predict(self, X):
        res = np.zeros(len(X))
        for i in range(len(self.models)):
            res += self.coef[i]*self.models[i].predict(X)
        return res

    def predict_proba(self, X):
        res = np.zeros(shape=(len(X),2))
        for i in range(len(self.models)):
            res += self.coef[i]*self.models[i].predict_proba(X)
        return res

def parse_dt(x):
    if not isinstance(x, str):
        return None
    elif len(x) == len('2010-01-01'):
        return datetime.datetime.strptime(x, '%Y-%m-%d')
    elif len(x) == len('2010-01-01 10:10:10'):
        return datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    else:
        return None

def safe(f):
    def inner(*args):
        try:
            return f(*args)
        except:
            return None
    return inner

def transform_datetime_features(df):
    datetime_columns = [
        col_name
        for col_name in df.columns
        if col_name.startswith('datetime')
    ]
    for col_name in datetime_columns:
        df[col_name] = df[col_name].apply(lambda x: parse_dt(x))
        df['number_weekday_{}'.format(col_name)] = df[col_name].apply(safe(lambda x: x.weekday()))
        df['number_month_{}'.format(col_name)] = df[col_name].apply(safe(lambda x: x.month))
        df['number_day_{}'.format(col_name)] = df[col_name].apply(safe(lambda x: x.day))
        df['number_hour_{}'.format(col_name)] = df[col_name].apply(safe(lambda x: x.hour))
        df['number_hour_of_week_{}'.format(col_name)] = df[col_name].apply(safe(lambda x: x.hour + x.weekday() * 24))
        df['number_minute_of_day_{}'.format(col_name)] = df[col_name].apply(safe(lambda x: x.minute + x.hour * 60))
    return df

class Feature():
    def fit(self, feature):
        self.name = feature.name

        if self.name == 'target':
            if feature.nunique() == 2:
                return

            self.ss = StandardScaler()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', DataConversionWarning)
                self.ss.fit(np.array(feature).reshape(-1,1))
            return

        if self.name.startswith('number'):
            feature = pd.to_numeric(feature, errors='coerce') # 'coerce' mean that invalid values will be set as NaN

        elif self.name.startswith('id'):
            feature = pd.to_numeric(feature, errors='coerce') # 'coerce' mean that invalid values will be set as NaN
            self.nan_replace = -1
            feature = feature.fillna(self.nan_replace)

        elif self.name.startswith('string'):
            self.nan_replace = ''
            feature = feature.fillna(self.nan_replace)
            self.le = LabelEncoder()
            feature = self.le.fit_transform(feature)

        self.ss = StandardScaler()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DataConversionWarning)
            self.ss.fit(np.array(feature).reshape(-1,1))

        if self.name.startswith('number'):
            self.nan_replace = self.ss.mean_[0]

    def transform(self, feature):
        if self.name == 'target':
            if feature.nunique() == 2:
                return feature

            with warnings.catch_warnings():
                warnings.simplefilter('ignore', DataConversionWarning)
                feature = self.ss.transform(np.array(feature).reshape(-1,1))
            return feature

        if self.name.startswith('number') or self.name.startswith('id'):
            feature = pd.to_numeric(feature, errors='coerce') # 'coerce' mean that invalid values will be set as NaN
            feature = feature.fillna(self.nan_replace)

        elif self.name.startswith('string'):
            feature = feature.fillna(self.nan_replace)
            unknown_values = list(set(feature.tolist()).difference(self.le.classes_))
            feature = feature.replace(unknown_values, ['' for _ in unknown_values])
            feature = self.le.transform(feature)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DataConversionWarning)
            feature = self.ss.transform(np.array(feature).reshape(-1,1))

        return feature

    def fit_transform(self, feature):
        self.name = feature.name

        if self.name == 'target':
            if feature.nunique() == 2:
                return feature

            self.ss = StandardScaler()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', DataConversionWarning)
                feature = self.ss.fit_transform(np.array(feature).reshape(-1,1))
            return feature


        if self.name.startswith('number'):
            feature = pd.to_numeric(feature, errors='coerce') # 'coerce' mean that invalid values will be set as NaN

        elif self.name.startswith('id'):
            feature = pd.to_numeric(feature, errors='coerce') # 'coerce' mean that invalid values will be set as NaN
            self.nan_replace = -1
            feature = feature.fillna(self.nan_replace)

        elif self.name.startswith('string'):
            self.nan_replace = ''
            feature = feature.fillna(self.nan_replace)
            self.le = LabelEncoder()
            feature = self.le.fit_transform(feature)

        self.ss = StandardScaler()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DataConversionWarning)
            self.ss.fit(np.array(feature).reshape(-1,1))

            if self.name.startswith('number'):
                self.nan_replace = self.ss.mean_[0]
                feature = feature.fillna(self.nan_replace)

            feature = pd.Series(data=self.ss.transform(np.array(feature).reshape(-1,1)).ravel(), name=self.name)

        return feature

    def target_inverse_transform(self, df_y):
        if df_y.nunique() > 2:
            df_y = pd.Series(data=self.ss.inverse_transform(np.array(df_y).reshape(-1,1).ravel()), name='target')
        return df_y

class Preprocessor():
    def __init__(self, features=None):
        if features:
            self.features = features
            self.used_columns= self.features.keys()

    def fit(self, df):
        # features from datetime
        df = transform_datetime_features(df)
        datetime_columns = [
                col_name
                for col_name in df.columns
                if col_name.startswith('datetime')
        ]
        df.drop(datetime_columns, axis=1, inplace=True)

        # drop constant features
        constant_columns = [
            col_name
            for col_name in df.columns
            if len(df[col_name].unique()) == 1
        ]
        df.drop(constant_columns, axis=1, inplace=True)

        df.drop(['line_id'], axis=1, inplace=True)

        # feature extraction
        self.features = {}
        for col_name in df.columns:
            feature = Feature()
            feature.fit(df[col_name])
            self.features[col_name] = feature

        # used columns
        self.used_columns = self.features.keys()

    def transform(self, df):
        # features from datetime
        df = transform_datetime_features(df)
        datetime_columns = [
                col_name
                for col_name in df.columns
                if col_name.startswith('datetime')
        ]
        df.drop(datetime_columns, axis=1, inplace=True)

        # drop non used columns
        df.drop(list(set(df.columns).difference(set(self.used_columns))), axis=1, inplace=True)

        # transform
        for col_name in df.columns:
            df[col_name] = self.features[col_name].transform(df[col_name])

        if 'target' in df.columns:
            df_y = df.target
            df_X = df.drop('target', axis=1)
            return df_X, df_y

        return df

    def fit_transform(self, df):
        # features from datetime
        df = transform_datetime_features(df)
        datetime_columns = [
                col_name
                for col_name in df.columns
                if col_name.startswith('datetime')
        ]
        df.drop(datetime_columns, axis=1, inplace=True)

        # drop constant features
        constant_columns = [
            col_name
            for col_name in df.columns
            if len(df[col_name].unique()) == 1
        ]
        df.drop(constant_columns, axis=1, inplace=True)

        df.drop(['line_id'], axis=1, inplace=True)

        # feature extraction
        self.features = {}
        for col_name in df.columns:
            feature = Feature()
            df[col_name] = feature.fit_transform(df[col_name])
            self.features[col_name] = feature

        # used columns
        self.used_columns = self.features.keys()

        if 'target' in df.columns:
            df_y = df.target
            df_X = df.drop('target', axis=1)
            return df_X, df_y

        return df

    def target_inverse_transform(self, df_y):
        df_y = self.features['target'].target_inverse_transform(df_y)
        return df_y
