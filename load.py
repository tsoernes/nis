from itertools import chain

import numpy as np
import pandas as pd

from load_utils import (convert_num_to_ocat_ip, convert_to_ocat_ip, get_inches,
                        no_nans, none_cat, none_to_nan_ip, none_unspec,
                        not_applicable, string_clean_ip)
from vartypes import (bin_cats, continuous_vars, no_none_unspec_cats,
                      none_unspec_cats, ordered_multi_cats,
                      unordered_multi_cats)

# TODO
# Figure out how to encode data for xboost (0 vs NaN)
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
# - Convert none_unspec -> NaN
# - Convert 0 -> NaN
# - XGB decides what to do with NaN
#
#
# XGB decides at training time whether missing values go into the right or left node.
# It chooses which to minimise loss.
# Note that the gblinear booster treats missing values as zeros.
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

    # 0 means missing data for this var
    df['MachineHoursCurrentMeter'].replace(0, np.nan, inplace=True)

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
        unique = no_nans(df[col].unique())
        assert len(unique) == 2
        print(f"{col} as ordered cat with unique {unique}")
        convert_to_ocat_ip(df, col, unique)

    # Non-ordered categories for >2 cats
    for col in unordered_multi_cats:
        df[col] = df[col].astype('category')

    # Create (numerically or a priory) ordered cats
    for col, otype in ordered_multi_cats.items():
        none_to_nan_ip(df, col)
        if type(otype) is list:
            convert_to_ocat_ip(df, col, otype)
        else:
            convert_num_to_ocat_ip(df, col, otype)

    # There should not be any none_unspec left in data set at this point
    for col in df.columns:
        unique = no_nans(df[col].unique())
        assert none_unspec not in unique, (col, unique)

    for col in continuous_vars:
        # 64 -> 32 bit. Could possibly set NaN to 0 and use int32
        df[col] = df[col].astype(np.float32)

    # One-hot encode non-ordered cats
    if oh:
        df = pd.get_dummies(df, columns=unordered_multi_cats, prefix='oh-')
        ohcats = filter(lambda c: c.startswith('oh-'), df.columns)
        for col in ohcats:
            # uint8 -> bool
            df[col] = df[col].astype(np.bool)

    # TODO Distribute NaN into cats according to their size
    # alternatively, set NaN for OH cats after conversion
    return df
