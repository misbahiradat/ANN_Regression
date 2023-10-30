

__all__ = ['CategoricalTransformer', 'PolynomialTransformer', 'categorical_encoding', 'drop_nan_zero_rows',
           'custom_scikit_train_test_split', 'custom_dask_train_test_split']

# Cell
#export
import pandas as pd
import dask.array as da
import numpy as np
from dask_ml.model_selection import train_test_split as dask_train_test_split
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures

# Cell
class CategoricalTransformer(BaseEstimator, TransformerMixin):

    # Initializing the categorical transformation
    def __init__(self, cat_encoding = 'ohe'):

        # Writing the type of transformer
        self.cat_encoding = cat_encoding

    # Fitting categorical transformer
    def fit(self, X, y = None):

        # Extracting categorical variables
        if isinstance(X, pd.DataFrame):
            cat_var_names = X.select_dtypes(['object', 'category']).columns.to_list()

        # Fitting categorical transformer
        if cat_var_names.__len__() > 0:

            self.cat_var_names = cat_var_names

            # One-hot encoding case
            if self.cat_encoding == 'ohe':

                # Getting the one hot encoding of categorical variables
                X_ohe = pd.get_dummies(X, self.cat_var_names)
                self.cat_encoding_info = X_ohe.columns

            # Mean-encoding case
            elif hasattr(self, 'cat_encoding') and self.cat_encoding == 'me':

                if y is None:
                    raise ValueError('Mean encoding requires a pandas series as the same length of X.')

                unique_categ_labels = {} # Dictionary to store unique entries in categorical variables
                X_me = X.copy().assign(**{y.name: y.values}) # Combining input and out
                for categ in self.cat_var_names:

                    # Mean encoding the variable
                    mean_encoded_subject = X_me.groupby(categ)[y.name].mean().to_dict()

                    # Unique entries in categorical variables
                    unique_categ_labels[categ] = mean_encoded_subject

                # Writing the unique categorical labels to class
                self.cat_encoding_info = unique_categ_labels

        return self

    # Transforming future datasets  based on the fit transformer
    def transform(self, X_t):

        if hasattr(self, 'cat_var_names'):

            # One-hot encoding case
            if self.cat_encoding == 'ohe':

                X_t = pd.get_dummies(X_t, self.cat_var_names)
                missing_cols = set(self.cat_encoding_info) - set(X_t.columns)
                for i in missing_cols:
                    X_t[i] = 0
                X_t = X_t[self.cat_encoding_info]

                return X_t

            # Mean-encoding case
            elif self.cat_encoding == 'me':

                # Casting the mean encoded subjects to test set
                X_tc = X_t.copy()
                for i in self.cat_encoding_info:
                    X_tc.loc[:, i] = X_t.loc[:, i].map(self.cat_encoding_info[i])

                return X_tc

        # Return the same input in case there is no categorical feature
        else:
            return X_t

    # Fit-transform
    def fit_transform():

        # Combined fit and transformation
        return self.fit(X, y).transform(X)


# Cell
class PolynomialTransformer(BaseEstimator, TransformerMixin):

    # Initialization of PolynomialTransformer class
    def __init__(self, PD):

        # Writing PD to polynomial transformer class
        self.PD = PD

        # Preparing the polynomial features class
        self.poly = PolynomialFeatures(np.max(PD), include_bias = False)

    # Fitting categorical transformer
    def fit(self, X):

        # Setting up polynomial degrees
        if isinstance(self.PD, (int, np.int64)):
            # If an integer is entered for PD, it is replicated to accommodate all variables
            self.PD = [self.PD] * X.shape[1]

        if self.PD.__len__() < X.shape[1]:
            raise ValueError('PD variable must either be an integer or a list/numpy array with'\
                             ' the same number of arrays as the number of variables in input.')

        # Raise error for negative values
        if any(n < 0 for n in self.PD):
            raise ValueError('Negative values for polynomials are not allowed.')

        # Maximum degree for polynomials
        self.poly.fit(np.zeros((1, X.shape[1])))

        # Removing the monomials with variables that have larger powers than
        # assigned in power array
        self.valid_poly_indices = (self.poly.powers_ <= np.tile(self.PD,(self.poly.powers_.shape[0],1))).all(1)
        self.PowerMatrix = self.poly.powers_[self.valid_poly_indices]

        return self

    # Transforming future datasets  based on the fit transformer
    def transform(self, X_t):

        X_t = self.poly.transform(X_t)
        return X_t[:, self.valid_poly_indices]

    # Fit-transform
    def fit_transform(self, X):

        # Combined fit and transformation
        return self.fit(X).transform(X)

# Cell
def categorical_encoding(df, categ_labels, output_label, cat_encoding = 'ohe'):

    # One-hot encoding
    if cat_encoding == 'ohe':

        # Getting the one hot encoding of categorical variables
        df = pd.get_dummies(df, categ_labels)

        return df, df.columns

    # Mean encoding
    elif cat_encoding == 'me':

        unique_categ_labels = {} # Dictionary to store unique entries in categorical variables
        for categ in categ_labels:

            # Mean encoding the variable
            mean_encoded_subject = df.groupby(categ)[output_label].mean().to_dict()

            # Unique entries in categorical variables
            unique_categ_labels[categ] = mean_encoded_subject

            # Mapping values to categorical variables
            df[categ] = df[categ].map(mean_encoded_subject)

        return df, unique_categ_labels

    # Raising error if an invalid option for cat_encoding is entered
    else:

        raise ValueError('cat_encoding parameter must be either "ohe" or "me".')

# Cell
def drop_nan_zero_rows(df, drop_nan = True, output_label = None):

    # Dropping rows with nans
    if isinstance(drop_nan, bool) and drop_nan:
        df = df.dropna().reset_index(drop = True).reset_index(drop = True)
    else:
        raise ValueError('drop_nan parameter must be either True or False.')

    # Dropping rows with zero in the output value
    if output_label is None: # No zero output row is dropped
        return df.reset_index(drop = True)
    elif output_label is not None and output_label in df.columns:
        return df[df[output_label] != 0].reset_index(drop = True)
    else:
        raise ValueError('output_label must be either None or a label from the df.columns')
    
# custom_train_test_split for dask base function
def custom_dask_train_test_split(x, y, 
                                 test_size=0.2, 
                                 split_type="randomsplit", 
                                 random_state=None):
    """
    This function is useful for splitting the data either randomly or sequentially for dask objets
    and it will aslo return the indices of the train and test data poitns
    """

    # decide random or sequential
    shuffle = True if split_type == "randomsplit" else False
 
    if shuffle: # random split
        X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(x, y,
                                                                                         da.arange(x.shape[0]),  
                                                                                         test_size=test_size,
                                                                                         shuffle=True,
                                                                                         random_state=random_state)
    else: # sequential split
        X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(x, y,
                                                                                         da.arange(x.shape[0]), 
                                                                                         test_size=test_size,
                                                                                         shuffle=False,
                                                                                         random_state=None)
    return X_train, X_test, y_train, y_test, train_indices, test_indices
    
# custom_train_test_split for sklearn base function
def custom_scikit_train_test_split(x, y, 
                            test_size=0.2, 
                            split_type="randomsplit", 
                            random_state=None):
    
    """
    This function is useful for splitting the data either randomly or sequentially for pandas or sklearn objects
    and it will aslo return the indices of the train and test data poitns
    """
    
    # decide random or sequential
    shuffle = True if split_type == "randomsplit" else False
 
    if shuffle: # random split
        X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(x, y,
                                                                                         np.arange(x.shape[0]),  
                                                                                         test_size=test_size,
                                                                                         shuffle=True,
                                                                                         random_state=random_state)
    else: # sequential split
        X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(x, y,
                                                                                         np.arange(x.shape[0]), 
                                                                                         test_size=test_size,
                                                                                         shuffle=False,
                                                                                         random_state=None)
    return X_train, X_test, y_train, y_test, train_indices, test_indices