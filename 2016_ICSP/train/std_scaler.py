from sklearn.preprocessing import StandardScaler

def get_scaler() :
    """Creates object with the sklearn.preprocessing.StandardScaler scaler"""
    return StandardScaler(copy = True, with_mean = True, with_std = True)