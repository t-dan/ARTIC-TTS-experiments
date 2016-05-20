from sklearn.base import TransformerMixin


class DummyScaler(TransformerMixin):
    """Dummy scaler - no transformaton of input data is done."""

    def __init__(self):
        pass

    # noinspection PyPep8Naming
    def transform(self, X, y=None):
        return X

    # noinspection PyPep8Naming
    def fit(self, X, y=None):
        return self

    # noinspection PyPep8Naming
    def fit_transform(self, X, y=None, **fit_params):
        return X


def get_scaler() :
    """Creates object with the dummy scaler"""
    return DummyScaler()