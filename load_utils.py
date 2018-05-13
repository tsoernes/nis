import re

import numpy as np
import openpyxl as op
import pandas as pd
from pandas.api.types import CategoricalDtype
from pandas.core.categorical import Categorical
from scipy.sparse import csr_matrix

not_applicable = 'Unspecified'
none_unspec = 'None or Unspecified'
none_cat = 'NoneCat'
nan_cat = 'NaNCat'


def cun(df, x, ac=False, vc=False):
    """ Helper function for browsing data set
    au: Print all unique
    vc: Print value counts"""
    if type(x) is int:
        print(df.iloc[:10, x])
        data = df.iloc[:, x]
    else:
        print(df[x][:10])
        data = df[x]
    if vc:
        print(data.value_counts(dropna=False))
    y = data.unique()
    last = None if ac else 10
    print(y[:last], y.shape)


def datadict_to_pandas(df):
    """ Get descriptions to a DF with same cols as dataset """
    wb = op.load_workbook("Data Dictionary.xlsx")
    desc_df = pd.DataFrame(wb['Data Dictionary'].values)
    desc_df.drop(index=0, inplace=True, axis=0)
    df.iloc[1] = df.iloc[1].astype(str).str.cat(df.iloc[2].astype(str))
    desc_df.drop(desc_df.columns[2], inplace=True, axis=1)
    desc_df = desc_df.transpose()
    # desc_df.columns = desc_df.iloc[0].str.lower()
    desc_df.drop(index=0, inplace=True, axis=0)
    return desc_df


def string_clean_ip(df, col):
    df[col] = df[col].str.strip().str.lower()


def isnan(e):
    try:
        return np.isnan(e)
    except TypeError:
        return False


def no_nans(iterable):
    return list(filter(lambda e: not isnan(e), iterable))


def none_to_nan_ip(df, col):
    df[col].replace(none_unspec, np.nan, inplace=True)


inch_re = re.compile(r"([0-9]+)\' ([0-9]*)\"")


def get_inches(el):
    if (type(el) is not str and np.isnan(el)) or el == none_unspec:
        return el
    m = inch_re.match(el)
    if m is None:
        raise Exception(el)
    else:
        return int(m.group(1)) * 12 + float(m.group(2))


def convert_num_to_ocat_ip(df, col, typ):
    """Ordered category for numeric types.
    NaNs have lowest order and should be
    interpreted as missing"""
    unique = df[col].unique()
    nn_unique = no_nans(unique)
    # Sort in natural order by type, e.g. int("1")
    order = sorted(nn_unique, key=typ)
    if len(unique) != len(nn_unique):
        df[col] = df[col].fillna(nan_cat)
        df[col] = df[col].astype('category')
        order = (nan_cat, *order)
        # print(df[col].cat.categories, df[col].unique(), order)
    convert_to_ocat_ip(df, col, order)


def convert_to_ocat_ip(df, col, order):
    """Ordered category. """
    cat_type = CategoricalDtype(categories=order, ordered=True)
    df[col] = df[col].astype(cat_type)


def sparse_dummies(categorical_values):
    categories = Categorical.from_array(categorical_values)
    N = len(categorical_values)
    row_numbers = np.arange(N, dtype=np.int)
    ones = np.ones((N, ))
    return csr_matrix((ones, (row_numbers, categories.codes)))


def sparse_dummies_from_codes(codes):
    N = len(codes)
    row_numbers = np.arange(N, dtype=np.int)
    ones = np.ones((N, ))
    return csr_matrix((ones, (row_numbers, codes.values)))
