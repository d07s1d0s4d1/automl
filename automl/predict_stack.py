import click
import os
import pandas as pd
import pickle
import time
import numpy as np
import catboost as cb
from sklearn.metrics import roc_auc_score

from utils import Preprocessor, Model

# use this to stop the algorithm before time limit exceeds
TIME_LIMIT = int(os.environ.get('TIME_LIMIT', 5*60))


@click.command()
@click.option('--test-csv', required=True)
@click.option('--test-target-csv', default = None, required=False)
@click.option('--prediction-csv', required=True)
@click.option('--model-dir', required=True)
def main(test_csv, test_target_csv, prediction_csv, model_dir):
    start_time = time.time()

    # load model
    model_config_filename = os.path.join(model_dir, 'model_config.pkl')
    metric_file = os.path.join(model_dir, 'rating.txt')

    with open(model_config_filename, 'rb') as fin:
        model_config = pickle.load(fin)

    # read dataset
    df = pd.read_csv(test_csv)
    print('Dataset read, shape {}'.format(df.shape))

    line_id = df['line_id']

    preprocessor = Preprocessor(model_config['features'])
    df_X = preprocessor.transform(df)

    model = model_config['model']

    if model_config['mode'] == 'regression':
        df['prediction'] = model.predict(df_X.values)

    elif model_config['mode'] == 'classification':
        df['prediction'] = model.predict_proba(df_X.values)[:, 1]

    df['line_id'] = line_id
    df[['line_id', 'prediction']].to_csv(prediction_csv, index=False)

    print('Prediction time: {}'.format(time.time() - start_time))

    if test_target_csv:
        def save_metric(metric):
            with open(metric_file, 'a') as f:
                f.write('{}\n'.format(metric))

        # read targets
        test = pd.read_csv(test_target_csv)

        print('Read targets, shape {}'.format(test.shape))

        if model_config['mode'] == 'regression':
            pred = preprocessor.target_inverse_transform(df['prediction'])
            mse = np.mean((test.target - pred)**2)
            r_2 = 1 - mse/np.std(test.target)**2
            print('MSE: {}'.format(mse))
            print('R^2: {}'.format(r_2))
            save_metric(r_2)
        elif model_config['mode'] == 'classification':
            auc = roc_auc_score(test.target, df['prediction'])
            print('AUC: {}'.format(auc))
            save_metric(auc)

if __name__ == '__main__':
    main()
