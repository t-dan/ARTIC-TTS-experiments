from sklearn.cross_validation import ShuffleSplit

def get_scaler(n_items) :
    """Creates object with the sklearn.cross_validation.ShuffleSplit data splitter.

       Args:
         n_items (int): Total number of elements in the training dataset.
    """
    return ShuffleSplit(n_items, n_iter = 10, test_size = 0.2)