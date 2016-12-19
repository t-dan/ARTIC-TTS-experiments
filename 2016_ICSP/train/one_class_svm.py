from sklearn.svm import OneClassSVM


def get_OCC() :
    """Creates object with OCSVM classifier"""
    return OneClassSVM(cache_size = 200, coef0 = 0.0, degree = 3, gamma = 0.0, kernel = 'rbf', max_iter = -1, nu = 0.5, random_state = None, shrinking = True, tol = 0.001, verbose = False)

def get_GridSearch():
    """Creates grid search parameters for OCSVM classifier training"""
    return { 'nu'   : (0.005,          0.01,           0.015,
                       0.02,           0.025,          0.05,
                       0.075,          0.1,            0.125,
                       0.15,           0.175,          0.2,
                       0.225,          0.25,           0.275,
                       0.3),
             'gamma': (3.05175781e-05, 6.10351562e-05, 1.22070312e-04,
                       2.44140625e-04, 4.88281250e-04, 9.76562500e-04,
                       0.00195312500,  0.00390625,     0.0078125,
                       0.015625,       0.03125,        0.0625,
                       0.125,          0.25,           0.5,
                       1.0,            2.0,            4.0,
                       8.0)
           }
