import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.metrics import root_mean_squared_error


def custom_predict(X, threshold, model):
        probs = model.predict_proba(X) 
        return (probs[:, 1] > threshold).astype(int)

def logistic_regression_fit_predict(
    df: pd.DataFrame, 
    target_column, 
    test_size=0.2, 
    random_state=42,
    threshold = None,
    penalty = 'l2', 
    return_model=False,
    print_metrics=True
):
    """
    Обучает модель логистической регрессии на данных и выводит метрики качества.

    Параметры:
    ----------
    df : pandas.DataFrame
        Датасет лишь с колонками, на которых будет обучаться модель + таргет
    target_column : str
        Название целевой переменной
    test_size : float, default=0.2
        Размер тестовой выборки (0.0 - 1.0).
    penalty: penalty: Literal['l1', 'l2', 'elasticnet'] | None = "l2",
        Штраф при обучении модели
    random_state : int, default=42
        Seed для воспроизводимости.
    return_model : bool, default=False
        Возвращать ли обученную модель и результаты.
    print_metrics : bool, default=True
        Печатать ли метрики качества.

    Возвращает:
    -----------
    model: LogisticRegression or None
        Если return_model=True, возвращает модель 
    """
    # Проверка данных
    if target_column not in df.columns:
        raise ValueError(f"Целевая колонка '{target_column}' не найдена в датафрейме.")
    
    if df[target_column].nunique() != 2:
        raise ValueError("Целевая переменная должна быть бинарной (2 уникальных значения).")

    X = df.drop(columns=target_column)
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # стандартизирует диапазон входных данных
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(penalty=penalty)
    model.fit(X_train_scaled, y_train)
    
    # сделать красиво определения трешхолда для регрессии по дизбалансу таргета
    target_dist = df[target_column].value_counts()
    target_probability = target_dist[1] / target_dist[0]
    if threshold is not None:
        target_probability = threshold
    
    predictions = custom_predict(X=X_test_scaled, threshold=target_probability, model=model)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Метрики
    if print_metrics:
        print("Метрики качества модели:")
        print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
        print(f"Precision: {precision_score(y_test, predictions):.4f}")
        print(f"Recall: {recall_score(y_test, predictions):.4f}")
        print(f"F1-score: {f1_score(y_test, predictions):.4f}")
        print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
        print(f"rmse: {root_mean_squared_error(y_test, y_proba):.4f}")
        print("\nМатрица ошибок:")
        print("tn    fn\nfp     tp:")
        print(confusion_matrix(y_test, predictions))
    
    if return_model:
        return model
    else:
        return None
    


def logistic_regression_fit_predict_coefs(
    df: pd.DataFrame, 
    target_column, 
    test_size=0.2, 
    random_state=42,
    threshold = None,
    penalty = 'l2', 
    return_model=False,
    print_metrics=True
):
    """
    Обучает модель логистической регрессии на данных и выводит метрики качества.

    Параметры:
    ----------
    df : pandas.DataFrame
        Датасет лишь с колонками, на которых будет обучаться модель + таргет
    target_column : str
        Название целевой переменной
    test_size : float, default=0.2
        Размер тестовой выборки (0.0 - 1.0).
    penalty: penalty: Literal['l1', 'l2', 'elasticnet'] | None = "l2",
        Штраф при обучении модели
    random_state : int, default=42
        Seed для воспроизводимости.
    return_model : bool, default=False
        Возвращать ли обученную модель и результаты.
    print_metrics : bool, default=True
        Печатать ли метрики качества.

    Возвращает:
    -----------
    model: LogisticRegression or None
        Если return_model=True, возвращает модель 
    """
    # Проверка данных
    if target_column not in df.columns:
        raise ValueError(f"Целевая колонка '{target_column}' не найдена в датафрейме.")
    
    if df[target_column].nunique() != 2:
        raise ValueError("Целевая переменная должна быть бинарной (2 уникальных значения).")

    X = df.drop(columns=target_column)
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # стандартизирует диапазон входных данных
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(penalty=penalty)
    model.fit(X_train_scaled, y_train)
    
    # сделать красиво определения трешхолда для регрессии по дизбалансу таргета
    target_dist = df[target_column].value_counts()
    target_probability = target_dist[1] / target_dist[0]
    if threshold is not None:
        target_probability = threshold
    
    predictions = custom_predict(X=X_test_scaled, threshold=target_probability, model=model)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    column_statistic = [(e1, e2) for e1, e2 in zip(df.columns, model.coef_[0])]
    
    # Метрики
    if print_metrics:
        print("Метрики качества модели:")
        print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
        print(f"Precision: {precision_score(y_test, predictions):.4f}")
        print(f"Recall: {recall_score(y_test, predictions):.4f}")
        print(f"F1-score: {f1_score(y_test, predictions):.4f}")
        print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
        print(f"rmse: {root_mean_squared_error(y_test, y_proba):.4f}")
        print("\nМатрица ошибок:")
        print("tn    fn\nfp     tp:")
        print(confusion_matrix(y_test, predictions))
    
    if return_model:
        return model, column_statistic
    else:
        return None, None