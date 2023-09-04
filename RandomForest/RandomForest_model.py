import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from common import grid_model_create, run_model, dump_best_estimator, scaler_model


def start_model(X_train, X_test, y_train, y_test):

    if isinstance(X_train, str):
        X_train = pd.read_csv(X_train)

    if isinstance(X_test, str):
        X_test = pd.read_csv(X_test)

    if isinstance(y_train, str):
        y_train = pd.read_csv(y_train)

    if isinstance(y_test, str):
        y_test = pd.read_csv(y_test)

    X_train_scaler, X_test_scaler, scaler = scaler_model(X_train, X_test)

    param_grid = {'n_estimators': [64, 80, 96, 112, 128],
                  'max_features': ['sqrt', 'log2'],
                  'bootstrap': [False, True]}

    grid_model = grid_model_create(RandomForestRegressor(), param_grid, 'neg_mean_squared_error')

    run_model(grid_model, X_train_scaler, X_test_scaler, y_train.values.ravel(), y_test.values.ravel())

    return grid_model, scaler


def save_model(model, scaler):
    dump_best_estimator(model.best_estimator_, 'ElasticNet', 'model')
    dump_best_estimator(scaler, 'ElasticNet', 'scaler')
