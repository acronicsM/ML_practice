import numpy as np
import pandas as pd
from common import preparation, split_data, save_preparation_date
from ElasticNet import ElasticNet_model
from PolynomialFeatures import PolynomialFeatures_model
from KNN import KNeighborsRegressor_model
from SVR import SVR_model
from RandomForest import RandomForest_model


if __name__ == "__main__":
    file_name = input('Укажите путь к файлу: ')
    target = input('Укажите имя целевой переменной: ')

    df = pd.read_csv(file_name)
    df.sample(100)
    print(f'Количество строк: {len(df)}')
    print(f'Average value "SalePrice": {np.mean(df[target])}')

    if input('Нужна ли разбить данные на обучающею и тестовую выборку (Y/N): ').upper() == 'Y':
        test_size = float(input('Укажите размер тестовой выборки (от 0 до 1): '))
        random_state = int(input('Укажите значение random_state (int): '))

        X_train, X_test, y_train, y_test = preparation(df, target, test_size, random_state)
        X_val = y_val = None

        if input('Нужна ли разбить тестовые данные выборку валидации и тестирования (Y/N): ').upper() == 'Y':
            test_size = float(input('Укажите размер выборки валидации (от 0 до 1): '))
            random_state = int(input('Укажите значение random_state (int): '))
            X_test, X_val, y_test, y_val = split_data(X_test, y_test, test_size, random_state)

        if input('Нужна ли сохранить данные разбивки (Y/N): ').upper() == 'Y':
            save_preparation_date(X_train, X_test, y_train, y_test, X_val, y_val)

    else:
        X_train = input('Укажите имя файла X_train: ')
        X_test = input('Укажите имя файла X_test: ')
        y_train = input('Укажите имя файла y_train: ')
        y_test = input('Укажите имя файла y_test: ')

    models = [ElasticNet_model, PolynomialFeatures_model, KNeighborsRegressor_model, SVR_model, RandomForest_model]

    menu = """
        Выберите модель прогнозирования (int)
        1. ElasticNet
        2. PolynomialFeatures
        3. KNN
        4. SVR
        5. RandomForest
    """

    answer = int(input(menu + '\n'))
    if answer < 1 or answer > len(models):
        exit('Не верный выбор')

    model = models[answer - 1]
    grid_model, scaler = model.start_model(X_train, X_test, y_train, y_test)

    print(grid_model.best_params_)

    if input('Нужна ли сохранить данные модели (Y/N): ').upper() == 'Y':
        model.save_model(grid_model.best_estimator_, scaler)
