import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from common import grid_model_create, run_model, dump_best_estimator


def start_model(X_train, X_test, y_train, y_test):

    if isinstance(X_train, str):
        X_train = pd.read_csv(X_train)

    if isinstance(X_test, str):
        X_test = pd.read_csv(X_test)

    if isinstance(y_train, str):
        y_train = pd.read_csv(y_train)

    if isinstance(y_test, str):
        y_test = pd.read_csv(y_test)

    pipe = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsRegressor())])
    param_grid = {'knn__n_neighbors': [30, 40, 50]}

    grid_model = grid_model_create(pipe, param_grid, 'neg_mean_squared_error', cv=5)

    run_model(grid_model, X_train, X_test, y_train, y_test)

    return grid_model, None


def save_model(model, scaler):
    dump_best_estimator(model.best_estimator_, 'KNN', 'model')
    dump_best_estimator(scaler, 'KNN', 'scaler')
