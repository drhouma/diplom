# import roles

import pandas as pd
from typing import Dict

from pandas import Series
from reader.roles import *
from typing import cast

import warnings
from warnings import filterwarnings
filterwarnings("ignore")

from datetime import datetime

def get_key_by_value(dictionary, value):
    for key, val in dictionary.items():
        if val == value:
            return key
    return None

class Reader:
    
    """
    Сопоставляет признакам их роли
    роли: {categorial, numeric}
    таргет ('target') роль и id роль должны быть указаны при инициализации
    """
    
    # dict with keys as feature names, values feature role
    roles = {}
    max_nan_rate = 0.5
    max_constant_rate = 0.995
    target_col = None
    
    data = pd.DataFrame
    
    def __init__(self, dataset: pd.DataFrame, target_col, id_col, max_constant_rate=None, max_nan_rate=None):
        if max_constant_rate is not None:
            self.max_constant_rate = max_constant_rate
        if max_nan_rate is not None:
            self.max_nan_rate = max_nan_rate
        self.data = dataset.copy()
        self.data = self.data.drop(columns=id_col, errors='ignore')
        self.target_col = target_col
        
        self.guess_roles()
        
    def guess_roles(self, columns = None):
        self.roles = {}
        self.roles[self.target_col] = TargetRole(self.data[self.target_col])
        if columns is None:
            columns = self.data.columns.values
        for column in columns:
            if column not in self.roles.keys():
                if self._is_ok_feature(self.data[column]):
                    self.roles[column] = self._guess_role(self.data[column])
                else:
                    self.roles[column] = DropRole()
        
        self.roles["Numeric"] = self.get_numeric_features()
        self.roles["Categorial"] = self.get_categorial_features()
    
    
    def _guess_role(self, feature: Series):
        """ Попытка узнать какой тип данных в признаке
        Если конвертируется в float - numeric
        Если конвертируется в дату - datetime
        Если не конвертируется - categorial
        """
        
        # check if feature is number
        try:
            _ = feature.astype(float)
            if feature.nunique(dropna=True) <= 2:
                return CategoryRole(object)
            return NumericRole(float)
        except ValueError:
            pass
        except TypeError:
            pass

        # check if it's datetime
        # dt_role = DatetimeRole(np.datetime64, date_format=date_format)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                t = cast(pd.Series, pd.to_datetime(feature, format=np.datetime64))
        except (ValueError, AttributeError, TypeError):
            # else category
            return CategoryRole(object)

    def _is_ok_feature(self, feature) -> bool:
        """ Проверка колонки, будет ли он нормальным признаком

        Args:
            feature: колонка из датасета

        Returns:
            ``False`` если много нанов или в признаке доминирует одно значение

        """
        if feature.isnull().mean() >= self.max_nan_rate:
            return False
        if (feature.value_counts().values[0] / feature.shape[0]) >= self.max_constant_rate:
            return False
        
        return True
    
    
    def _get_features_role(self, Role):
        columns = []
        for key in self.roles:
            # print(type(self.roles[key].name))
            if key not in ["Numeric", "Categorial"]:
                if self.roles[key] == Role:
                    columns.append(key)
        return columns
    
    def get_drop_role_features(self):
        return self._get_features_role(DropRole())
    
    def get_numeric_features(self):
        return self._get_features_role(NumericRole())
    
    def get_categorial_features(self):
        return self._get_features_role(CategoryRole())