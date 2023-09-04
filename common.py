import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import dump


def run_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    print(f'RMSE: {np.sqrt(mean_squared_error(y_test, pred))}\t\tMAE: {mean_absolute_error(y_test, pred)}')


def scaler_model(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaler = scaler.fit_transform(X_train)
    X_test_scaler = scaler.transform(X_test)

    return X_train_scaler, X_test_scaler, scaler


def grid_model_create(model, param_grid, scoring, cv=10, verbose=2):
    grid_model = GridSearchCV(estimator=model,
                              param_grid=param_grid,
                              scoring=scoring,  # стратегия определения лучшей модели
                              cv=cv,  # на сколько частей разбивать данные
                              verbose=verbose,  # уровень логов
                              )

    return grid_model


def dump_best_estimator(data, name_model, name_data):
    dump(data, f'termokit_order_{name_model}_{name_data}.joblib')


def results(model: GridSearchCV):
    print(f'Optimal Hyperparams: {model.best_params_}')
    means, stds = model.cv_results_['mean_test_score'], model.cv_results_['std_test_score']

    for mean, std, params in zip(means, stds, model.cv_results_['params']):
        print(f'Mean: {round(mean,2)}   Standar Devition: {round(std*2,2)}    Hyperparameters{params}')


def preparation(df_data: pd.DataFrame, target: str, test_size=0.3, random_state=42):
    no_none = ['Hour', 'Website', 'InitialQuantity', 'InitialCost', 'SaleAmount']

    new_df = df_data[:]
    new_df.PaymentMethod = new_df.PaymentMethod.fillna('Наличными')
    new_df[no_none] = new_df[no_none].fillna(0)

    df_nums, df_objs = new_df.select_dtypes(exclude='object'), new_df.select_dtypes(include='object')
    df_objs = pd.get_dummies(df_objs, drop_first=True)
    new_df = pd.concat([df_nums, df_objs], axis=1)

    X, y = new_df.drop(labels=target, axis=1), new_df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test


def split_data(X: pd.DataFrame, y: pd.DataFrame, test_size, random_state=42):
    X_test, X_val, y_test, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_test, X_val, y_test, y_val


def save_preparation_date(X_train, X_test, y_train, y_test, X_val=None, y_val=None):
    X_train.to_csv('X_train.csv', index=False)
    X_test.to_csv('X_test.csv', index=False)
    if X_val:
        X_val.to_csv('X_val.csv', index=False)

    y_train.to_csv('y_train.csv', index=False)
    y_test.to_csv('y_test.csv', index=False)
    if y_val:
        y_val.to_csv('y_val.csv', index=False
                     )