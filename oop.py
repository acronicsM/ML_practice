from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from joblib import dump
from sklearn.linear_model import ElasticNet


class MyModel:

    MODEL = MODEL_NAME = None

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()

    def scaler_model(self, x_train):
        return self.scaler.fit_transform(x_train)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def save_model(self):
        dump(self.model.best_estimator_, f'{self.MODEL_NAME}_model.joblib')
        dump(self.scaler, f'{self.MODEL_NAME}_scaler.joblib')


class ElasticNetModel(MyModel):

    MODEL = ElasticNet
    MODEL_NAME = 'ElasticNet'

    def __init__(self):
        super().__init__()
        self.__ml_model = self.MODEL(max_iter=1_000_000)

    def fit(self, x_train, y_train, param_grid, scoring, cv=10, verbose=2):

        x_train_scaler = self.scaler_model(x_train)

        grid_model = GridSearchCV(estimator=self.__ml_model,
                                  param_grid=param_grid,
                                  scoring=scoring,  # стратегия определения лучшей модели
                                  cv=cv,  # на сколько частей разбивать данные
                                  verbose=verbose,  # уровень логов
                                  )

        grid_model.fit(x_train_scaler, y_train)

        self.model = grid_model

