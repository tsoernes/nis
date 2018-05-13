import numpy as np
import pandas as pd
from scipy.sparse import hstack

from load_utils import (convert_num_to_ocat_ip, convert_to_ocat_ip, get_inches,
                        nan_cat, no_nans, none_cat, none_to_nan_ip,
                        none_unspec, not_applicable, sparse_dummies,
                        string_clean_ip)
from vartypes import (bin_cats, continuous_vars, no_none_unspec_cats,
                      none_unspec_cats, ordered_multi_cats,
                      unordered_multi_cats)

# TODO
# Figure out metric
# Parse fiProductClassDesc
# Reintroduce datetime
#
# Handling features with missing values.
# Notation:
# XGB - XGBoost
# cat - category
# var - variable
# catvar - categorical variable
# none_unspec - 'None or Unspecified'
# cat counts excludes NaN and none_unspec
#
# (A) For vars with 1 cat:
# - Any catvar must have 2 cats. Create cat N for none_unspec
# - Order cats (X, N)
# - XGB decides what to do with NaN
#
# (B) For vars with 2 cats (X < Y) and no none_unspec:
# - Order cats (X, Y)
# - XGB decides what to do with NaN
#
# (C) For vars with 2 cats and none_unspec or >2 cats:
# - Create cat N for none_unspec
# - One-Hot encode cats
# - Propagate NaNs to OH cats
#
# (D) For vars with >2 ordered cats:
# - Convert none_unspec -> NaN
# - XGB decides what to do with NaN
#
# (E) For continuous vars:
# - Convert none_unspec, NaN -> 0
# - XGB decides what to do with 0
#
#
# XGB decides at training time whether missing values go into the right or left node.
# It chooses which to minimise loss.
# Note that the gblinear booster treats missing values as zeros.
#
# Example statistics:
# Maximum machine hours
# In [1889]: df['MachineHoursCurrentMeter'].max()
# Out[1889]: 2483300.0
#
# Mean machine hours
# In [1910]: df['MachineHoursCurrentMeter'][df['MachineHoursCurrentMeter'] > 0].mean()
# Out[1910]: 7053.819247997828
#
# In [1946]: df['Stick_Length'].value_counts(dropna=False)
# Out[1949]:
# NaN                    310437
# None or Unspecified     81539
# 9' 6"                    5832
# 10' 6"                   3519
# 11' 0"                   1601
# 9' 10"                   1463
# 9' 8"                    1462
# 9' 7"                    1423
# 12' 10"                  1087
# 10' 2"                   1004
# 8' 6"                     908
# 8' 2"                     614
# 10' 10"                   414
# 12' 8"                    322
# 11' 10"                   307
# 8' 4"                     274
# 8' 10"                    104
# 12' 4"                    103
# 9' 5"                     101
# 15' 9"                     87
# 6' 3"                      51
# 13' 7"                     11
# 13' 9"                      7
# 14' 1"                      7
# 13' 10"                     7
# 19' 8"                      5
# 7' 10"                      3
# 15' 4"                      3
# 24' 3"                      2
# 9' 2"                       1
# Name: Stick_Length, dtype: int64
#
#
#
#
#


def load_prep(oh=False):
    """oh: One-hot encode some catvars"""

    df = pd.read_pickle('trainvalid-wdates.pkl')

    # fiModelDesc has disaggregations
    # SalesID is unique
    # fiProductClassDes has _some_ disaggregations
    #
    columns = ['fiModelDesc', 'SalesID', 'fiProductClassDesc']
    df.drop(columns, inplace=True, axis=1)

    string_columns = ['fiBaseModel', 'fiSecondaryDesc']
    map(string_clean_ip, string_columns)

    col = 'fiBaseModel'
    df[col] = df[col].str.replace('-', '')

    col = 'fiModelSeries'
    df[col] = df[col].str.replace('-', '')
    srcs = ['SeriesII', '6.00E+00', '7.00E+00']
    targ = ['II', '6', '7']
    df[col].replace(srcs, targ, inplace=True)
    # Maybe replace II -> 2 etc

    col = 'fiModelDescriptor'
    srcs = ['2.00E+00', '3.00E+00', '7.00E+00']
    targ = ['2', '3', '7']
    df[col].replace(srcs, targ, inplace=True)

    col = 'Enclosure'
    df[col].replace('EROPS w AC', 'EROPS AC', inplace=True)

    col = 'YearMade'
    df[col].replace(1000, 0, inplace=True)

    col = 'Drive_System'
    df[col].replace('All Wheel Drive', 'Four Wheel Drive', inplace=True)

    col = 'Blade_Width'
    df[col] = df[col].str.replace("'", '')
    df[col].replace('<12', '1', inplace=True)

    col = 'Tire_Size'
    df[col] = df[col].str.replace('"', '')
    df[col].replace('10 inch', '10', inplace=True)

    col = 'Undercarriage_Pad_Width'
    df[col] = df[col].str.replace(" inch", '')

    df["Stick_Length"] = df["Stick_Length"].apply(get_inches)

    col = 'MachineHoursCurrentMeter'
    # float64 -> int32
    df[col] = df[col].fillna(0)
    df[col] = df[col].astype(np.int32)

    # Just store year of sale in order to
    # save some memory and training time
    df['saledate'] = df['saledate'].apply(lambda d: d.year)
    for col in df.columns:
        if df[col].dtype == np.int64:
            df[col] = df[col].astype(np.int32)
    # Got to fit in memory ..
    # df = df.drop(df.index[:100000])

    # Make none_unspec cat
    for col in none_unspec_cats:
        unique = no_nans(df[col].unique())
        assert none_unspec in unique, (col, unique)
        df[col].replace(none_unspec, none_cat, inplace=True)

    for col in no_none_unspec_cats:
        unique = no_nans(df[col].unique())
        assert none_unspec not in unique, (col, unique)

    # Create ordered category for binary vars
    for col in bin_cats:
        unique = df[col].unique()
        nn_unique = no_nans(unique)
        assert len(nn_unique) == 2
        if len(unique) == len(nn_unique):
            order = unique
        else:
            # Put NaN as middle ordered cat. See:
            # http://fastml.com/converting-categorical-data-into-numbers-with-pandas-and-scikit-learn/
            order = (nn_unique[0], nan_cat, nn_unique[1])
        print(f"{col} as ordered cat with order {order}")
        convert_to_ocat_ip(df, col, order)
        df[col] = df[col].fillna(nan_cat)

    # Non-ordered categories for >2 cats
    for col in unordered_multi_cats:
        df[col] = df[col].astype('category')

    # Create (numerically or a priory) ordered cats
    for col, otype in ordered_multi_cats.items():
        none_to_nan_ip(df, col)
        if type(otype) is list:
            convert_to_ocat_ip(df, col, (nan_cat, *otype))
        else:
            convert_num_to_ocat_ip(df, col, otype)
        unique = df[col].unique()
        nn_unique = no_nans(unique)
        if len(unique) != len(nn_unique):
            df[col] = df[col].fillna(nan_cat)

    # for col in df.columns:
    #     # There should not be any none_unspec left in data set at this point
    #     unique = df[col].unique()
    #     nn_unique = no_nans(unique)
    #     assert none_unspec not in nn_unique, (col, nn_unique)
    #     if df[col].dtype.name == "category":
    #         has_nan_cat = nan_cat in df[col].cat.categories
    #         # Use pd-internal int8 cat coding
    #         df[col] = df[col].values.codes
    #         # We only want NaN-cats to have code 0,
    #         # which is interpreted as missing by XGB
    #         codes = unique.codes
    #         for i, cat in enumerate(unique):
    #             # assert not ((codes[i] == 0) ^ (cat == nan_cat)), (col, unique, codes)
    #             if (codes[i] == 0) and (cat != nan_cat) and col != 'YearMade':
    #                 # df[col] = df[col] + 1

    #         has_nan = not len(unique) == len(nn_unique)
    #         codes = df[col].unique()
    #         has_nan_code = 0 in codes
    #         print(
    #             f"\n{col} NanVals:{has_nan}, NanCat:{has_nan_cat}, NanCode:{has_nan_code}"
    #             f"\nCats:{unique}\nCodes:{codes}")

    # One-hot encode non-ordered cats
    if oh:
        print("Constructing one-hots")
        # hstack(sparse_dummies_from_codes(df[col]) for col in df.columns), format="csr")
        prefixes = list(map(lambda s: "oh-" + s, unordered_multi_cats))
        df = pd.get_dummies(df, columns=unordered_multi_cats, prefix=prefixes)
        # df[col] = df[col].astype(np.bool)
        df = df.to_sparse(fill_value=0)
        return df

    return df


def save_oh():
    df = load_prep(oh=True)
    df.to_pickle("trainvalid-oh.pkl")


# if __name__ == "__main__":
#     load_prep(oh=False)
