import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from reader.reader import Reader
from ml_selectors.permutation_importance import feature_selection_permutation_importance
from ml_selectors.rfe import rank_features_with_rfe
from models.logistic_reg import logistic_regression_fit_predict, logistic_regression_fit_predict_coefs
from models.random_forest import optimize_random_forest
from transformers.categorial import impute_with_mode, one_hot_encode_columns
from transformers.numeric import handle_outliers, fill_numeric_missing, normalize_numeric_features


def auto_ml_pipeline(
    data: pd.DataFrame,
    target_column: str,
    id_column: str,
    nan_rate_range: list = [0.3, 0.5, 0.7],
    const_rate_range: list = [0.9, 0.95, 0.99],
    outliers_threshold: float = 3.5,
    outliers_percentage_threshold: float = 0.05,
    target_probability = None,
    random_state: int = 42
):
    """
    Автоматический ML пайплайн с обработкой данных, выбросов и подбором моделей.
    
    Параметры:
    ----------
    data : pd.DataFrame
        Исходный датасет
    target_column : str
        Название целевой переменной
    id_column : str
        Название колонки с ID
    nan_rate_range : list, optional
        Диапазон значений для подбора max_nan_rate
    const_rate_range : list, optional
        Диапазон значений для подбора max_constant_rate
    outliers_threshold : float, optional
        Порог для обнаружения выбросов (в стандартных отклонениях)
    outliers_percentage_threshold : float, optional
        Порог процента выбросов для выбора стратегии обработки
    random_state : int, optional
        Seed для воспроизводимости
        
    Возвращает:
    -----------
    dict
        Словарь с результатами: лучшие параметры, метрики и модели
    """
    # 1. Подбор оптимальных параметров для Reader
    best_f1 = -1
    best_params = {}
    if target_probability is None:
        target_dist = data[target_column].value_counts(ascending=True)
        target_probability = target_dist[1] / target_dist[0]
    
    

    
    for nan_rate in nan_rate_range:
        for const_rate in const_rate_range:
            # Инициализация Reader с текущими параметрами
            rd = Reader(
                dataset=data,
                target_col=target_column,
                id_col=id_column,
                max_nan_rate=nan_rate,
                max_constant_rate=const_rate
            )
            rd.data = impute_with_mode(df=rd.data, columns=rd.roles['Numeric'], strategy='median')

            rd.data = impute_with_mode(df=rd.data, columns=rd.roles['Categorial'], strategy='most_frequent')
            rd.guess_roles()
            rd.data = one_hot_encode_columns(df=rd.data, columns_to_encode=rd.roles['Categorial'])
            rd.guess_roles()
            
            # Подготовка данных
            X = rd.data[rd.roles['Numeric'] + rd.roles['Categorial']]
            y = rd.data[target_column]
            
            # Разделение на train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=random_state
            )
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Обучение базовой логистической регрессии
            model = LogisticRegression(random_state=random_state)
            model.fit(X_train_scaled, y_train)
            
            # Оценка качества
            probs = model.predict_proba(X_test_scaled)
            y_pred = (probs[:, 1] > target_probability).astype(int)
            
            current_f1 = f1_score(y_test, y_pred)
            
            # Сохранение лучших параметров
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_params = {
                    'max_nan_rate': nan_rate,
                    'max_constant_rate': const_rate,
                    'f1_score': current_f1
                }
    
    print(f"Лучшие параметры Reader: {best_params}")
    
    # 2. Инициализация Reader с лучшими параметрами
    rd = Reader(
        dataset=data,
        target_col=target_column,
        id_col=id_column,
        max_nan_rate=best_params['max_nan_rate'],
        max_constant_rate=best_params['max_constant_rate']
    )
    
    # 3. Обработка пропущенных значений
    # Для числовых признаков
    numeric_cols = rd.roles['Numeric']
    rd.data = impute_with_mode(df=rd.data, columns=rd.roles['Numeric'], strategy='median')
    
    # Для категориальных признаков
    categorial_cols = rd.roles["Categorial"]
    if categorial_cols:
        rd.data = impute_with_mode(
            rd.data,
            columns=categorial_cols,
            strategy='most_frequent',
            copy=False
        )
    
    
    # 4. Обработка выбросов в два этапа
    # Этап 1: Обработка явных выбросов (zscore > 3.5)
    if numeric_cols:
        # Сначала обрабатываем только явные выбросы
        rd.data, _ = handle_outliers(
            rd.data,
            columns=numeric_cols,
            method='zscore',
            strategy='impute',
            threshold=outliers_threshold,
            verbose=False
        )
    
    # Этап 2: Определение стратегии для остальных выбросов
    if numeric_cols:
        outliers_stat = {}
        res = {'transform': [], 'impute': [], 'clip': []}
        
        for col in numeric_cols:
            # Расчет выбросов по IQR
            q1 = rd.data[col].quantile(0.25)
            q3 = rd.data[col].quantile(0.75)
            iqr_range = q3 - q1
            lower_bound = q1 - 1.5 * iqr_range
            upper_bound = q3 + 1.5 * iqr_range
            
            outliers_mask = (rd.data[col] < lower_bound) | (rd.data[col] > upper_bound)
            outliers_count = outliers_mask.sum()
            outliers_percentage = outliers_count / len(rd.data)
            outliers_stat[col] = outliers_percentage
            
            # Расчет дисперсии выбросов
            outliers_variance = rd.data[col][outliers_mask].var()
            overall_variance = rd.data[col].var()
            
            # Выбор стратегии
            if outliers_variance > 2 * overall_variance:
                strategy = 'transform'
            else:
                strategy = 'clip'
            
            res[strategy].append(col)
        
        # Применение стратегий
        if res['transform']:
            rd.data, _ = handle_outliers(
                rd.data,
                columns=res['transform'],
                method='iqr',
                strategy='transform',
                threshold=1.5,
                verbose=False
            )
        
        if res['clip']:
            rd.data, _ = handle_outliers(
                rd.data,
                columns=res['clip'],
                method='iqr',
                strategy='clip',
                threshold=1.5,
                verbose=False
            )
    
    # 5. Кодирование категориальных переменных
    rd.guess_roles()
    categorial_cols = rd.roles['Categorial']
    if categorial_cols:
        rd.data = one_hot_encode_columns(
            rd.data,
            columns_to_encode=categorial_cols,
            drop='first',
            sparse=False
        )
    rd.guess_roles()
    
    # 6. Отбор признаков
    X = rd.data[rd.roles["Numeric"] + rd.roles['Categorial']]
    y = rd.data[target_column]
    
    # Используем RFE для первичного отбора
    ranked_features = rank_features_with_rfe(
        data=rd.data[rd.roles["Numeric"] + rd.roles['Categorial'] + [target_column]],
        target_column=target_column,
        estimator=LogisticRegression(random_state=random_state),
        n_features_to_select=1,
        verbose=False
    )
    
    # Берем топ-N признаков
    N = len(ranked_features) / 2
    top_features = [f for (rank, f) in ranked_features if rank <= N]
    X = X[top_features]
    
    # 7. Подбор threshold для логистической регрессии
    best_threshold = None
    best_f1_lr = -1
    best_model_lr = None
    
    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    
    # Масштабирование данных
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Перебор threshold
    for threshold in np.linspace(target_probability / 2, target_probability * 2, 10):
        model_lr = LogisticRegression(random_state=random_state)
        model_lr.fit(X_train_scaled, y_train)
        # Кастомный predict с threshold
        probs = model_lr.predict_proba(X_test_scaled)
        y_pred = (probs[:, 1] > threshold).astype(int)
        
        current_f1 = f1_score(y_test, y_pred)
        
        if current_f1 > best_f1_lr:
            best_f1_lr = current_f1
            best_threshold = threshold
            best_model_lr = model_lr
    
    print(f"Лучший threshold для LR: {best_threshold:.2f}, F1: {best_f1_lr:.4f}")
    
    # 8. Обучение Random Forest с оптимизацией гиперпараметров
    best_model_rf, best_params_rf, best_score_rf = optimize_random_forest(
        X=X,
        y=y,
        n_trials=20,
        cv_folds=3,
        scoring='f1',
        random_state=random_state,
        n_jobs=-1
    )
    
    # 9. Оценка моделей на тестовом наборе
    # Для логистической регрессии
    probs_lr = best_model_lr.predict_proba(X_test_scaled)
    y_pred_lr = (probs_lr[:, 1] > best_threshold).astype(int)
    
    # Для случайного леса
    y_pred_rf = best_model_rf.predict(X_test)
    
    # Метрики
    metrics = {
        'LogisticRegression': {
            'accuracy': accuracy_score(y_test, y_pred_lr),
            'precision': precision_score(y_test, y_pred_lr),
            'recall': recall_score(y_test, y_pred_lr),
            'f1': f1_score(y_test, y_pred_lr),
            'roc_auc': roc_auc_score(y_test, probs_lr[:, 1])
        },
        'RandomForest': {
            'accuracy': accuracy_score(y_test, y_pred_rf),
            'precision': precision_score(y_test, y_pred_rf),
            'recall': recall_score(y_test, y_pred_rf),
            'f1': f1_score(y_test, y_pred_rf),
            'roc_auc': roc_auc_score(y_test, best_model_rf.predict_proba(X_test)[:, 1])
        }
    }
    
    # 10. Возвращаем результаты
    return {
        'best_reader_params': best_params,
        'best_threshold': best_threshold,
        'best_rf_params': best_params_rf,
        'models': {
            'LogisticRegression': best_model_lr,
            'RandomForest': best_model_rf
        },
        'metrics': metrics,
        'selected_features': top_features,
        'preprocessed_data': rd.data
    }

