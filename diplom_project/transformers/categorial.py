import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer




def impute_with_mode(df, columns=None, strategy='most_frequent', copy=True):
    """
    Заменяет пропущенные значения на наиболее часто встречающиеся (моду) в указанных колонках.
    
    Параметры:
    ----------
    df : pandas.DataFrame
        Входной DataFrame с пропущенными значениями
    columns : list, optional
        Список колонок для обработки (по умолчанию все колонки)
    strategy : str, default 'most_frequent'
        Стратегия импутации ('most_frequent', 'mean', median)
    copy : bool, default True
        Если True, возвращает копию DataFrame, не изменяя оригинал
    
    Возвращает:
    -----------
    pandas.DataFrame
        DataFrame с заполненными пропусками
    """
    if columns is None:
        return df

    if copy:
        df = df.copy()
    
    # Если колонки не указаны, берем все
    
    # Создаем и применяем импутер
    imputer = SimpleImputer(strategy=strategy)
    
    for col in columns:
        if df[col].isna().any():  # Проверяем наличие пропусков
            # Преобразуем в 2D массив для импутера
            df[col] = imputer.fit_transform(df[[col]]).ravel()
    
    return df

def one_hot_encode_columns(df, columns_to_encode, drop='first', sparse=False, handle_unknown='ignore'):
    """
    Трансформирует выбранные колонки датасета с помощью OneHotEncoder.
    
    Параметры:
    ----------
    df : pandas.DataFrame
        Исходный датасет
    columns_to_encode : list
        Список колонок для OneHot кодирования
    drop : {'first', 'if_binary', None} или array-like, default='first'
        Параметр для удаления одной из колонок (чтобы избежать дамми-ловушки)
    sparse : bool, default=False
        Возвращать ли разреженную матрицу (для экономии памяти)
    handle_unknown : {'error', 'ignore'}, default='ignore'
        Как обрабатывать неизвестные категории при трансформации
    
    Возвращает:
    -----------
    pandas.DataFrame
        Новый датасет с закодированными колонками
    """
    # Создаем копию датасета, чтобы не изменять исходный
    df_encoded = df.copy()
    
    # Инициализируем OneHotEncoder
    encoder = OneHotEncoder(drop=drop, sparse_output=sparse, handle_unknown=handle_unknown)
    
    # Применяем к выбранным колонкам
    encoded_data = encoder.fit_transform(df_encoded[columns_to_encode])
    
    # Создаем DataFrame с закодированными признаками
    encoded_df = pd.DataFrame(
        encoded_data,
        columns=encoder.get_feature_names_out(columns_to_encode),
        index=df_encoded.index
    )
    
    # Удаляем исходные колонки и добавляем закодированные
    df_encoded = df_encoded.drop(columns=columns_to_encode)
    df_encoded = pd.concat([df_encoded, encoded_df], axis=1)
    
    return df_encoded

def ordinal_encode_columns(df, columns_to_encode, categories='auto', handle_unknown='use_encoded_value', unknown_value=0):
    """
    Трансформирует выбранные колонки датасета с помощью OrdinalEncoder.
    
    Параметры:
    ----------
    df : pandas.DataFrame
        Исходный датасет
    columns_to_encode : list
        Список колонок для ordinal кодирования
    categories : 'auto' или list of array-like, default='auto'
        Категории для каждого признака ('auto' - определяются из данных)
    handle_unknown : {'error', 'use_encoded_value'}, default='use_encoded_value'
        Как обрабатывать неизвестные категории при трансформации
    unknown_value : int, default=-1
        Значение для неизвестных категорий (если handle_unknown='use_encoded_value')
        
    Возвращает:
    -----------
    pandas.DataFrame
        Новый датасет с закодированными колонками
    """
    # Создаем копию датасета, чтобы не изменять исходный
    df_encoded = df.copy()
    
    # Инициализируем OrdinalEncoder
    encoder = OrdinalEncoder(
        categories=categories,
        handle_unknown=handle_unknown,
        unknown_value=unknown_value
    )
    
    # Применяем к выбранным колонкам
    df_encoded[columns_to_encode] = encoder.fit_transform(df_encoded[columns_to_encode])
    
    return df_encoded