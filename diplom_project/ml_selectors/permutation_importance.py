import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def feature_selection_permutation_importance(
    X, 
    y, 
    model=None, 
    n_repeats=4, 
    random_state=42, 
    n_jobs=-1,
    plot_top_n=20,
    threshold=None
):
    """
    Отбирает признаки с помощью Permutation Feature Importance.
    
    Параметры:
    ----------
    X : pandas.DataFrame или numpy.ndarray
        Матрица признаков.
    y : pandas.Series или numpy.ndarray
        Целевая переменная.
    model : sklearn estimator, optional
        Модель для оценки важности (по умолчанию LogisticRegression).
    n_repeats : int, default=4
        Количество перестановок для каждого признака.
    random_state : int, default=42
        Seed для воспроизводимости.
    n_jobs : int, default=-1
        Количество ядер для параллельных вычислений.
    plot_top_n : int, optional
        Количество признаков для визуализации (None - отключить график).
    threshold : float, optional
        Порог важности для отбора признаков (None - возвращает все).
        
    Возвращает:
    -----------
    tuple : (important_features, importance_df)
        - Список важных признаков (если threshold указан)
        - DataFrame с важностью всех признаков
    """
    # Разделение данных
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    
    # Используем RandomForest по умолчанию
    if model is None:
        raise "No model is specified"
    
    # Вычисление важности признаков
    result = permutation_importance(
        model, 
        X_val, 
        y_val, 
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=n_jobs,
        scoring='accuracy'
    )
    
    # Создание DataFrame с результатами
    importance_df = pd.DataFrame({
        'feature': X.columns if hasattr(X, 'columns') else [f'feature_{i}' for i in range(X.shape[1])],
        'importance_mean': result.importances_mean,
        'importance_std': result.importances_std
    }).sort_values('importance_mean', ascending=False)
    
    # Отбор признаков по порогу
    if threshold is not None:
        important_features = importance_df[importance_df['importance_mean'] > threshold]['feature'].tolist()
        return important_features, importance_df
    else:
        return None, importance_df