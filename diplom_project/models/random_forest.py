import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import optuna

def optimize_random_forest(
    X, 
    y, 
    n_trials=50, 
    cv_folds=5, 
    scoring='accuracy', 
    random_state=42,
    n_jobs=-1,
    show_progress_bar=True
):
    """
    Оптимизирует гиперпараметры Random Forest с помощью Optuna.
    
    Параметры:
    ----------
    X : pandas.DataFrame или numpy.ndarray
        Матрица признаков.
    y : pandas.Series или numpy.ndarray
        Целевая переменная (бинарная или многоклассовая).
    n_trials : int, default=50
        Количество итераций оптимизации.
    cv_folds : int, default=5
        Количество фолдов кросс-валидации.
    scoring : {'accuracy', 'rmse', 'roc_auc', 'f1'}, default='accuracy'
        Метрика для оптимизации.
    random_state : int, default=42
        Seed для воспроизводимости.
    n_jobs : int, default=-1
        Количество ядер для параллельных вычислений.
    show_progress_bar : bool, default=True
        Показывать progress bar Optuna.
        
    Возвращает:
    -----------
    tuple : (best_model, best_params, best_score)
        Лучшая модель, параметры и значение метрики.
    """
    # Проверка метрики
    valid_metrics = ['accuracy', 'rmse', 'roc_auc', 'f1']
    if scoring not in valid_metrics:
        raise ValueError(f"Неверная метрика. Допустимые значения: {valid_metrics}")

    # Целевая функция для Optuna
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 2, 10),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'random_state': random_state,
            'n_jobs': n_jobs,
            'class_weight': 'balanced'
        }
        
        model = RandomForestClassifier(**params)
        
        # Кросс-валидация
        kfold = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        scores = cross_val_score(
            model, X, y, cv=kfold, 
            scoring=scoring, n_jobs=n_jobs
        )
        
        return np.mean(scores)

    # Оптимизация
    study = optuna.create_study(direction='maximize')
    study.optimize(
        objective, 
        n_trials=n_trials, 
        show_progress_bar=show_progress_bar
    )

    # Лучшие параметры
    best_params = study.best_params
    best_params.update({
        'random_state': random_state,
        'n_jobs': n_jobs,
        'class_weight': 'balanced'
    })

    # Обучение финальной модели
    best_model = RandomForestClassifier(**best_params)
    best_model.fit(X, y)

    # Оценка на кросс-валидации
    kfold = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(
        best_model, X, y, cv=kfold, 
        scoring=scoring, n_jobs=n_jobs
    )
    best_score = np.mean(cv_scores)

    print(f"\nЛучшие параметры: {best_params}")
    print(f"Лучшее {scoring} (CV): {best_score:.4f}")

    return best_model, best_params, best_score