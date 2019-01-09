import argparse
import os
import pandas as pd
import pickle
import time
import numpy as np
from sklearn.metrics import roc_auc_score

from utils import transform_datetime_features

# use this to stop the algorithm before time limit exceeds
TIME_LIMIT = int(os.environ.get('TIME_LIMIT', 5*60))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-csv', type=argparse.FileType('r'), required=True)
    parser.add_argument('--test-target-csv', type=argparse.FileType('r'), default = None, required=False)
    parser.add_argument('--prediction-csv', type=argparse.FileType('w'), required=True)
    parser.add_argument('--model-dir', required=True)
    args = parser.parse_args()

    start_time = time.time()

    # load model
    model_config_filename = os.path.join(args.model_dir, 'model_config.pkl')
    with open(model_config_filename, 'rb') as fin:
        model_config = pickle.load(fin)

    # read dataset
    df = pd.read_csv(args.test_csv)
    print('Dataset read, shape {}'.format(df.shape))


    if not model_config['is_big']:
        # features from datetime
        df = transform_datetime_features(df)

        # categorical encoding
        for col_name, unique_values in model_config['categorical_values'].items():
            for unique_value in unique_values:
                df['onehot_{}={}'.format(col_name, unique_value)] = (df[col_name] == unique_value).astype(int)

    # missing values
    if model_config['missing']:
        df.fillna(-1, inplace=True)
    elif any(df.isnull()):
        df.fillna(value=df.mean(axis=0), inplace=True)

    # filter columns
    used_columns = model_config['used_columns']

    # scale
    X_scaled = model_config['scaler'].transform(df[used_columns])

    model = model_config['model']
    if model_config['mode'] == 'regression':
        df['prediction'] = model.predict(X_scaled)
    elif model_config['mode'] == 'classification':
        df['prediction'] = model.predict_proba(X_scaled)[:, 1]



    print('Prediction time: {}'.format(time.time() - start_time))

    if args.test_target_csv:
        # read targets
        tg = pd.read_csv(args.test_target_csv)

        print('Read targets, shape {}'.format(tg.shape))

        if model_config['mode'] == 'regression':
            mse = np.mean((tg.target - df.prediction)**2)
            r_2 = 1 - mse/np.std(tg.target)**2
            print('MSE: {}'.format(mse))
            print('R^2: {}'.format(r_2))
        elif model_config['mode'] == 'classification':
            auc = roc_auc_score(tg.target, df.prediction)
            print('AUC: {}'.format(auc))
