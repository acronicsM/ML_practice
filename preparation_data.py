import pandas as pd
from sklearn.model_selection import train_test_split


def preparation(df_data: pd.DataFrame, target: str, test_size=0.3, random_state=42, validation_size=0):
    no_none = ['Hour', 'Website', 'InitialQuantity', 'InitialCost', 'SaleAmount']

    new_df = df_data[:]
    new_df.PaymentMethod = new_df.PaymentMethod.fillna('Наличными')
    new_df[no_none] = new_df[no_none].fillna(0)

    df_nums, df_objs = new_df.select_dtypes(exclude='object'), new_df.select_dtypes(include='object')
    df_objs = pd.get_dummies(df_objs, drop_first=True)
    new_df = pd.concat([df_nums, df_objs], axis=1)

    x, y = new_df.drop(labels=target, axis=1), new_df[target]

    x_val = y_val = None

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

    if validation_size > 0:
        x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=test_size, random_state=random_state)

    return x_train, x_test, x_val, y_train, y_test, y_val


def save_preparation_date(x_train, x_test, y_train, y_test, x_val=None, y_val=None):
    x_train.to_csv('X_train.csv', index=False)
    x_test.to_csv('X_test.csv', index=False)
    if x_val:
        x_val.to_csv('X_val.csv', index=False)

    y_train.to_csv('y_train.csv', index=False)
    y_test.to_csv('y_test.csv', index=False)
    if y_val:
        y_val.to_csv('y_val.csv', index=False)
