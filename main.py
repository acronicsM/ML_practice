import numpy as np
import pandas as pd
from preparation_data import preparation
from models import ElasticNetModel, KNNModel, SVRModel, RandomForestModel
from sklearn.metrics import mean_squared_error, mean_absolute_error


def run_elastic_net_model(xtrain, xtest, ytrain, ytest):
    param_grid = {'alpha': [.1, .2, .3, .4, .5, .6, .7, .8, .9],
                  'l1_ratio': [.3, .4, .5, .6, .7, .8, .9]}

    enm = ElasticNetModel(max_iter=1_000_000)
    enm.fit(x_train=xtrain, y_train=ytrain, param_grid=param_grid)

    y_predict = enm.predict(xtest)
    print(f'RMSE: {np.sqrt(mean_squared_error(ytest, y_predict))}\t\tMAE: {mean_absolute_error(ytest, y_predict)}')


def run_knn_model(xtrain, xtest, ytrain, ytest):
    param_grid = {'knn__n_neighbors': [30, 40, 50]}

    knn = KNNModel()
    knn.fit(x_train=xtrain, y_train=ytrain, param_grid=param_grid)

    y_predict = knn.predict(xtest)
    print(f'RMSE: {np.sqrt(mean_squared_error(ytest, y_predict))}\t\tMAE: {mean_absolute_error(ytest, y_predict)}')


def run_svr_model(xtrain, xtest, ytrain, ytest):
    param_grid = {'C': [400, 500, 600],
                  'kernel': ['linear', 'rbf', 'poly'],
                  'gamma': ['scale', 'auto'],
                  'degree': [1, 2],
                  'epsilon': [6_000, 7_000, 8_000]}

    svr = SVRModel()
    svr.fit(x_train=xtrain, y_train=ytrain, param_grid=param_grid)

    y_predict = svr.predict(xtest)
    print(f'RMSE: {np.sqrt(mean_squared_error(ytest, y_predict))}\t\tMAE: {mean_absolute_error(ytest, y_predict)}')


def run_fandom_forest_model(xtrain, xtest, ytrain, ytest):
    param_grid = {'n_estimators': [64, 80, 96, 112, 128],
                  'max_features': ['sqrt', 'log2'],
                  'bootstrap': [False, True]}

    rfm = RandomForestModel()
    rfm.fit(x_train=xtrain, y_train=y_train, param_grid=param_grid)

    y_predict = rfm.predict(xtest)
    print(f'RMSE: {np.sqrt(mean_squared_error(ytest, y_predict))}\t\tMAE: {mean_absolute_error(ytest, y_predict)}')


if __name__ == "__main__":
    file_name = r'order_analysis2023_08_30.csv'
    target = 'SaleAmount'

    df = pd.read_csv(file_name)
    df = df.sample(100)
    print(f'Количество строк: {len(df)}')
    print(f'Average value "SalePrice": {np.mean(df[target])}')

    x_train, x_test, x_val, y_train, y_test, y_val = preparation(df, target, test_size=3, random_state=101)

    run_elastic_net_model(x_train, x_test, y_train, y_test)
    # run_knn_model(x_train, x_test, y_train, y_test)
    # run_svr_model(x_train, x_test, y_train, y_test)
    # run_fandom_forest_model(x_train, x_test, y_train, y_test)
