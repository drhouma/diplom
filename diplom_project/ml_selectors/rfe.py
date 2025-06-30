from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from operator import itemgetter

def rank_features_with_rfe(
    data, 
    target_column='TARGET', 
    estimator=None, 
    n_features_to_select=1,
    scale_features=True,
    random_state=42,
    test_size=0.2,
    verbose=True
):
    """
    Ранжирует признаки с помощью метода RFE (Recursive Feature Elimination)
    
    Параметры:
    ----------
    data : pd.DataFrame
        Исходный датасет с признаками и целевой переменной
    target_column : str, optional
        Название целевой колонки (по умолчанию 'TARGET')
    estimator : object, optional
        Базовый классификатор/регрессор (по умолчанию LogisticRegression())
    n_features_to_select : int, optional
        Количество признаков для отбора (по умолчанию 1)
    scale_features : bool, optional
        Нужно ли стандартизировать признаки (по умолчанию True)
    random_state : int, optional
        Seed для воспроизводимости (по умолчанию 42)
    test_size : float, optional
        Размер тестовой выборки (по умолчанию 0.2)
    verbose : bool, optional
        Выводить ли информацию о ранжировании (по умолчанию True)
    
    Возвращает:
    -----------
    list of tuples
        Список кортежей в формате (ранк, название_признака), отсортированный по важности
    """
    
    # Создаем копию данных и разделяем на X и y
    df = data.copy()
    X = df.drop(columns=[target_column], errors='ignore')
    y = df[target_column]
    
    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Стандартизация (если нужно)
    if scale_features:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test
    
    # Инициализация модели (по умолчанию - логистическая регрессия)
    if estimator is None:
        estimator = LogisticRegression(random_state=random_state)
    
    # Выполнение RFE
    rfe = RFE(estimator, n_features_to_select=n_features_to_select)
    rfe.fit(X_train_scaled, y_train)
    
    # Формирование списка с ранжированием
    features = X_train.columns.to_list()
    ranked_features = sorted(zip(rfe.ranking_, features), key=itemgetter(0))
    
    # Вывод результатов (если нужно)
    if verbose:
        print("Ранжирование признаков:")
        for rank, feature in ranked_features:
            print(f"{rank}: {feature}")
    
    return ranked_features