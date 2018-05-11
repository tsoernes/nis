import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

from load_utils import (convert_num_to_ocat_ip, convert_to_ocat_ip, get_inches,
                        no_nans, none_to_nan_ip, none_unspec, not_applicable,
                        string_clean_ip)
from vartypes import (continuous_vars, nan_as_cat, none_as_ocat,
                      ordered_multi_cats, unordered_cats)

# TODO
# Figure out how to encode data for xboost (0 vs NaN)
# Figure out metric
# Parse fiProductClassDesc
# Reintroduce datetime
#
## Handling features with missing values.
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
# (Bn) For vars with 2 cats (X, Y),
# where the possibility of a unknown third cat seems likely
# - Create cat N for none_unspec
# - Ordered cats (X, N, Y)
# - XGB decides what to do with NaN
#
# (Bu) For vars with 2 cats (X, Y),
# where the possibility of a unknown third cat seems very unlikely
# - Convert none_unspec -> NaN
# - Ordered cats (X, Y)
# - XGB decides what to do with NaN
#
# (C) For vars with >2 ordered cats:
# - Convert none_unspec -> NaN
# - XGB decides what to do with NaN
#
# (D) For remaining unordered cat vars:
# - Create cat for none_unspec
# - Distribute NaN into cats according to their size
# - One-hot encode cats
# - Do the same at test time
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

df = pd.read_csv('TrainAndValid.csv', na_values=not_applicable)
# df = pd.read_csv('TrainAndValid.csv', parse_dates=['saledate'], na_values=not_applicable)

# fiModelDesc has disaggregations
# SalesID is unique
# fiProductClassDes has _some_ disaggregations
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

# (Bu) to cat
for col in nan_as_cat:
    # Convert none_unspec -> NaN
    none_to_nan_ip(df, col)
    df[col] = df[col].astype('category')

# (Bo)
for col in none_as_ocat:
    # Create cat N for none_unspec
    # Ordered cats
    unique = no_nans(df[col].unique())
    unique_nn = list(filter(lambda x: x != none_unspec, unique))
    if len(unique_nn) == 1:
        order = [unique_nn[0], none_unspec]
    elif len(unique_nn) == 2:
        order = [unique_nn[0], none_unspec, unique_nn[1]]
    else:
        raise Exception((col, unique))
    print(order)
    cat_type = CategoricalDtype(categories=order, ordered=True)
    df[col] = df[col].astype(cat_type)

# (C)
for col, otype in ordered_multi_cats.items():
    none_to_nan_ip(df, col)
    if type(otype) is list:
        convert_to_ocat_ip(df, col, otype)
    else:
        convert_num_to_ocat_ip(df, col, otype)

# (D)
# TODO Distribute NaN into cats according to their size
# One-hot encode cats
for col in unordered_cats:
    df[col].replace(none_unspec, np.nan, inplace=True)
# unordered_cats_oh = pd.get_dummies(df, unordered_cats)
