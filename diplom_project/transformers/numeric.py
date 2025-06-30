import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from scipy.stats import zscore, iqr

def handle_outliers(
    data: pd.DataFrame,
    columns=None,
    method='iqr',   
    strategy='clip',
    threshold=1.5,
    verbose=True
):
    """
    Обрабатывает выбросы в числовых данных различными методами.

    Параметры:
    ----------
    data : pd.DataFrame
        Входной DataFrame с данными
    columns : list, optional
        Список колонок для обработки
    method : str, default 'iqr'
        Метод обнаружения выбросов:
        - 'iqr' - межквартильный размах
        - 'zscore' - стандартные отклонения
    strategy : str, default 'clip'
        Стратегия обработки выбросов:
        - 'clip' - заменяет на граничные значения
        - 'impute' - заменяет на медиану/среднее
        - 'transform' - применяет логарифмическое преобразование
    threshold : float, default 1.5
        Порог для методов IQR/Z-Score (множитель для IQR, кол-во σ для Z-Score)
    contamination : float, default 0.05
        Доля ожидаемых выбросов для Isolation Forest
    n_neighbors : int, default 20
        Количество соседей для LOF
    verbose : bool, default True
        Выводить информацию о количестве обработанных выбросов

    Возвращает:
    -----------
    pd.DataFrame
        DataFrame с обработанными выбросами
    """
    if columns is None:
        return df
    
    df = data.copy()
    outliers_stat= {}
    
    
    for col in columns:
        if col not in df.columns:
            continue
            
        outliers_mask = np.zeros(len(df), dtype=bool)
        
        # Обнаружение выбросов
        if method == 'iqr':
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr_range = q3 - q1
            lower_bound = q1 - threshold * iqr_range
            upper_bound = q3 + threshold * iqr_range
            outliers_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            
        elif method == 'zscore':
            z_scores = zscore(df[col])
            outliers_mask = np.abs(z_scores) > threshold
            
        else:
            raise ValueError(f"Неизвестный метод: {method}. Доступные методы: 'iqr', 'zscore'")
        
        # Обработка выбросов
        if strategy == 'clip' and method in ['iqr', 'zscore']:
            if method == 'iqr':
                df[col] = np.clip(df[col], lower_bound, upper_bound)
            else:  # zscore
                median = df[col].median()
                std = df[col].std()
                df[col] = np.clip(df[col], median - threshold*std, median + threshold*std)
                
        elif strategy == 'impute':
            if df[col].skew() > 1:  # Если распределение скошено
                impute_value = df[col].median()
            else:
                impute_value = df[col].mean()
            df.loc[outliers_mask, col] = impute_value
            
        elif strategy == 'transform':
            if (df[col] <= 0).any():
                # Если есть отрицательные значения, используем сдвиг
                shift = -df[col].min() + 1
                df[col] = np.log(df[col] + shift)
            else:
                df[col] = np.log1p(df[col])
                
        else:
            raise ValueError(f"Неизвестная стратегия: {strategy}. Доступные стратегии: 'clip', 'remove', 'impute', 'transform'")
        
        outliers_stat[col] = outliers_mask.sum() / len(df)
        if verbose:
            n_outliers = outliers_mask.sum()
            if n_outliers > 0:
                print(f"Колонка '{col}': обработано {n_outliers} выбросов ({n_outliers/len(df)*100:.2f}%)")
    
    return df, outliers_stat

def fill_numeric_missing(
    df, 
    columns_to_fill=None, 
    strategy='mean', 
    inplace=False
):
    """
    Заменяет пропущенные значения в числовых колонках на указанное значение, среднее или медиану.

    Параметры:
    ----------
    df : pandas.DataFrame
        Исходный датафрейм.
    columns_to_fill : list, optional
        Список числовых колонок для обработки (по умолчанию — все числовые колонки).
    strategy : {'mean', 'median', 'constant'}, default='mean'
        Стратегия заполнения:
        - 'mean' – среднее значение,
        - 'median' – медиана,
    inplace : bool, default=False
        Если True, изменяет исходный датафрейм, иначе возвращает копию.

    Возвращает:
    -----------
    pandas.DataFrame or None
        Если inplace=False, возвращает новый датафрейм с заполненными пропусками.
        Если inplace=True, возвращает None (изменяет исходный датафрейм).
    """
    if not inplace:
        df = df.copy()

    # Если колонки не указаны, берём все числовые
    if columns_to_fill is None:
        columns_to_fill = df.select_dtypes(include=np.number).columns.tolist()
        
    # проверить колонки на совпадение типа roles
    
    for col in columns_to_fill:
        if col not in df.columns:
            continue

        # Вычисляем значение для замены
        if strategy == 'mean':
            fill_val = df[col].mean()
        elif strategy == 'median':
            fill_val = df[col].median()
        else:
            raise ValueError(f"Недопустимая стратегия: {strategy}. Используйте 'mean', 'median' или 'constant'.")

        # Заполняем пропуски
        df[col].fillna(fill_val, inplace=True)
    
    return None if inplace else df


def normalize_numeric_features(
    df, 
    columns_to_normalize=None, 
    method='standard', 
    return_scaler=False,
    **scaler_kwargs
):
    """
    Нормализует указанные числовые колонки в датафрейме.

    Параметры:
    ----------
    df : pandas.DataFrame
        Исходный датафрейм.
    columns_to_normalize : list, optional
        Список колонок для нормализации (по умолчанию все числовые колонки).
    method : {'standard', 'minmax', 'robust', 'custom'}, default='standard'
        Метод нормализации:
        - 'standard' - StandardScaler (z-score: mean=0, std=1)
        - 'minmax' - MinMaxScaler (диапазон [0, 1])
        - 'robust' - RobustScaler (устойчивый к выбросам)
        - 'custom' - ручное указание mean/std или min/max
    return_scaler : bool, default=False
        Возвращать ли объект scaler для обратного преобразования.
    **scaler_kwargs : dict
        Дополнительные параметры для scaler:
        - Для 'custom': mean/std или min/max в виде словаря {col: value}

    Возвращает:
    -----------
    pandas.DataFrame or tuple
        Нормализованный датафрейм. Если return_scaler=True, возвращает (df, scaler).

    Примеры:
    --------
    >>> df_normalized = normalize_numeric_features(df, method='minmax')
    >>> df_normalized, scaler = normalize_numeric_features(df, return_scaler=True)
    >>> df_custom = normalize_numeric_features(
    ...     df, 
    ...     method='custom', 
    ...     mean={'age': 30}, 
    ...     std={'age': 10}
    ... )
    """
    df = df.copy()
    
    # Определение колонок для нормализации
    # check roles[f].name == 'Numeric'

    # Выбор метода нормализации
    scaler = None
    if method == 'standard':
        scaler = StandardScaler(**scaler_kwargs)
    elif method == 'minmax':
        scaler = MinMaxScaler(**scaler_kwargs)
    elif method == 'robust':
        scaler = RobustScaler(**scaler_kwargs)
    elif method == 'custom':
        # Ручная нормализация по заданным параметрам
        for col in columns_to_normalize:
            if 'mean' in scaler_kwargs and 'std' in scaler_kwargs:
                mean = scaler_kwargs['mean'].get(col, df[col].mean())
                std = scaler_kwargs['std'].get(col, df[col].std())
                df[col] = (df[col] - mean) / std
            elif 'feature_range' in scaler_kwargs:
                min_val = scaler_kwargs['min'].get(col, df[col].min())
                max_val = scaler_kwargs['max'].get(col, df[col].max())
                df[col] = (df[col] - min_val) / (max_val - min_val)
            else:
                raise ValueError("Для 'custom' укажите mean/std или min/max")
        return df if not return_scaler else (df, None)
    else:
        raise ValueError(f"Неизвестный метод нормализации: {method}")

    # Применение нормализации
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
    
    return df if not return_scaler else (df, scaler)