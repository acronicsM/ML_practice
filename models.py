from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from joblib import dump
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor


class MyModel:
    def __init__(self, model_name, model, **kwargs):
        self.model = model(**kwargs)
        self.scaler = StandardScaler()
        self.model_name = model_name
        self.trained_model = None

    def scaler_model(self, x_train):
        return self.scaler.fit_transform(x_train)

    def predict(self, x_test):
        return self.trained_model.predict(self.scaler.transform(x_test))

    def save_model(self):
        dump(self.trained_model, f'{self.model_name}_model.joblib')
        dump(self.scaler, f'{self.model_name}_scaler.joblib')

    def fit(self, x_train, y_train, param_grid, scoring='neg_mean_squared_error', cv=10, verbose=2):
        x_train_scaler = self.scaler_model(x_train)

        grid_model = GridSearchCV(estimator=self.model,
                                  param_grid=param_grid,
                                  scoring=scoring,  # стратегия определения лучшей модели
                                  cv=cv,  # на сколько частей разбивать данные
                                  verbose=verbose,  # уровень логов
                                  )

        grid_model.fit(x_train_scaler, y_train)

        self.trained_model = grid_model


class ElasticNetModel(MyModel):
    def __init__(self, **kwargs):
        super().__init__(model_name='ElasticNet', model=ElasticNet, **kwargs)


class KNNModel(MyModel):
    def __init__(self, **kwargs):
        model = Pipeline([('scaler', StandardScaler), ('knn', KNeighborsRegressor(**kwargs))])
        super().__init__(model_name='KNeighborsRegressor', model=model)


class SVRModel(MyModel):
    def __init__(self, **kwargs):
        super().__init__(model_name='SVR', model=SVR, **kwargs)


class RandomForestModel(MyModel):
    def __init__(self, **kwargs):
        super().__init__(model_name='RandomForest', model=RandomForestRegressor, **kwargs)
