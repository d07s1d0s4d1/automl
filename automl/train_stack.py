import click
import os
import numpy as np
import pandas as pd
import pickle
import time
import catboost as cb
import numbers

from sklearn.linear_model import Ridge, LogisticRegression
from utils import Preprocessor, Model

# use this to stop the algorithm before time limit exceeds
TIME_LIMIT = int(os.environ.get('TIME_LIMIT', 5*60))
ONEHOT_MAX_UNIQUE_VALUES = 20
BIG_DATASET_SIZE = 500 * 1024 * 1024


@click.command()
@click.option('--train-csv', required=True)
@click.option('--model-dir', required=True)
@click.option('--mode', type=click.Choice(['classification', 'regression']), required=True)
def main(train_csv, model_dir, mode):
    start_time = time.time()

    #df = pd.read_csv(args.train_csv, low_memory = False)
    df = pd.read_csv(train_csv)
    is_big = df.memory_usage().sum() > BIG_DATASET_SIZE


    # dict with data necessary to make predictions
    model_config = {}
    model_config['is_big'] = is_big

    preprocessor = Preprocessor()
    df_X, df_y = preprocessor.fit_transform(df)

    model_config['features'] = preprocessor.features

    print('Dataset read, shape {}'.format(df_X.shape))

    # fitting
    model_config['mode'] = mode
    if mode == 'regression':
        ridge_model = Ridge()

        cb_model = cb.CatBoostRegressor(iterations=300,
                                     boosting_type=('Ordered' if len(df_X) < 1000 else 'Plain'),
                                     od_type="IncToDec",
                                     depth=6,
                                     od_pval=0.0001,
                                     #learning_rate=0.03,
                                     loss_function='RMSE')
        models = [ridge_model, cb_model]
    else:
        log_reg_model = LogisticRegression()

        cb_model = cb.CatBoostClassifier(iterations=300,
                                      boosting_type=('Ordered' if len(df_X) < 1000 else 'Plain'),
                                      od_type="IncToDec",
                                      depth=6,
                                      od_pval=0.0001,
                                      #learning_rate=0.03,
                                      loss_function='Logloss',
                                      logging_level='Verbose')
        models = [log_reg_model, cb_model]


    for model in models:
        model.fit(df_X, df_y)


    D = [1/np.std(model.predict(df_X) - df_y)**2 for model in models]
    s = sum(D)
    coef = [d/s for d in D]

    model = Model(models, coef)

    model_config['model'] = model

    model_config_filename = os.path.join(model_dir, 'model_config.pkl')
    with open(model_config_filename, 'wb') as fout:
        pickle.dump(model_config, fout, protocol=pickle.HIGHEST_PROTOCOL)

    print('Train time: {}'.format(time.time() - start_time))

if __name__ == '__main__':
    main()
