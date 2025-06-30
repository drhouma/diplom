 #!/usr/bin/env python -W ignore::DeprecationWarning
from AutoML.base import auto_ml_pipeline
import pandas as pd
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

if __name__ == "__main__":
    # Загрузка данных (пример)
    data = pd.read_csv('../data/application_train.csv')
    target_column = 'TARGET'
    id_column = 'SK_ID_CURR'
    data['DAYS_EMPLOYED'] = data['DAYS_EMPLOYED'].abs()
    # Запуск пайплайна
    results = auto_ml_pipeline(
        data=data,
        nan_rate_range=[0.3, 0.4, 0.5],
        const_rate_range=[0.995],
        target_column=target_column,
        id_column=id_column
    )
    
    # Вывод результатов
    print("\nМетрики моделей:")
    print("Логистическая регрессия:")
    print(results['metrics']['LogisticRegression'])
    print("\nСлучайный лес:")
    print(results['metrics']['RandomForest'])
    print(len(results['selected_features']))
    