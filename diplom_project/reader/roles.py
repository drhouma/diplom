import numpy as np

class ColumnRole:
    """ Абстрактный класс роли
    Тип роли содержит в себе доп информацию о том, какие 
    методы применять к ней
    """

    dtype = object
    force_input = False
    _name = "Abstract"

    @property
    def name(self) -> str:
        """Get str role name.

        Returns:
            str role name.

        """
        return self._name

    def __repr__(self) -> str:

        params = [(x, self.__dict__[x]) for x in self.__dict__ if x not in ["dtype", "name"]]

        return "{0} role, dtype {1}. Additional params: {2}".format(self.name, self.dtype, params)
    
    def __eq__(self, other) -> bool:
        return self.name == other.name



    @staticmethod
    def from_string(name: str) -> "ColumnRole":
        """Create default params role from string.

        Args:
            name: Role name.
            kwargs: Other parameters.

        Returns:
            Corresponding role object.

        """
        name = name.lower()

        if name in ["target"]:
            return TargetRole()

        if name in ["numeric"]:
            return NumericRole()

        if name in ["category"]:
            return CategoryRole()

        if name in ["datetime"]:
            return DatetimeRole()

        raise ValueError("Unknown string role: {}".format(name))


class NumericRole(ColumnRole):
    """ Числовой признак

    Args:
        dtype: тип переменной
        force_input: использовать признак при обучении, независимо от селектора
        prob: Если числа внутри - вероятность
        discretization: флаг дискретности данных

    """

    _name = "Numeric"

    def __init__(
        self,
        dtype = np.float32,
        force_input: bool = False,
        prob: bool = False,
        discretization: bool = False,
    ):
        self.dtype = dtype
        self.force_input = force_input
        self.prob = prob
        self.discretization = discretization


class CategoryRole(ColumnRole):
    """Категориальный тип данных

    Args:
        dtype: тип данных
        encoding_type: тип кодировки данных
        unknown: порог для интерпретации мало представленных категорий как пропущенных
        force_input: использовать признак при обучении, независимо от селектора
    Note:
        Корректные типы кодировок:

            - `'int'` - encode with int
            - `'freq'` - frequency encoding
            - `'ohe'` - one hot encoding

    """

    _name = "Category"

    def __init__(
        self,
        dtype = object,
        encoding_type: str = "freq",
        unknown: int = 5,
        force_input: bool = False,
        label_encoded: bool = False,
        ordinal: bool = False,
    ):
        self.dtype = dtype
        self.encoding_type = encoding_type
        self.unknown = unknown
        self.force_input = force_input
        self.label_encoded = label_encoded
        self.ordinal = ordinal


class DatetimeRole(ColumnRole):

    _name = "Datetime"

    def __init__(
        self,
        dtype = np.datetime64,
        
    ):
        self.dtype = dtype
        

class TargetRole(ColumnRole):
    """Таргет роль

    Args:
        dtype: тип данных роли
    """

    _name = "Target"

    def __init__(self, dtype = np.float32):
        self.dtype = dtype



class DropRole(ColumnRole):
    """Drop role."""

    _name = "Drop"
