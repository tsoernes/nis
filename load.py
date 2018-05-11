import numpy as np
import pandas as pd

from load_utils import (convert_num_to_ocat_ip, convert_to_ocat_ip, get_inches,
                        none_to_nan_ip, none_unspec, not_applicable,
                        string_clean_ip)
from vartypes import assume_no, continuous_vars, ordered_cats, unordered_cats

# TODO
# Figure out how to encode data for xboost (0 vs NaN)
# Figure out metric
# Parse fiProductClassDesc
# Reintroduce datetime

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
srcs = ['SeriesII' '6.00E+00' '7.00E+00']
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

df["Stick_Length"] = df["Stick_Length"].apply(get_inches)

for col in assume_no:
    df[col].replace(none_unspec, 'No', inplace=True)

assume_missing = set(df.columns) - set(assume_no)
for col in assume_missing:
    none_to_nan_ip(col)

for col, otype in ordered_cats.items():
    if type(otype) is list:
        convert_to_ocat_ip(df, col, otype)
    else:
        convert_num_to_ocat_ip(df, col, otype)

# one-hot encoding for non-ordered categorical vars
# unordered_cats_oh = pd.get_dummies(df, unordered_cats)

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
# (Bo) For vars with 2 ordered cats (X < Y),
# (Bu) or for unordered vars with 2 cats where the
# possibility of a unknown third cat seems unlikely
# - Create cat N for none_unspec
# - Ordered cats (X, N, Y)
# - XGB decides what to do with NaN
#
# (C) For vars with >2 ordered cats:
# - Convert none_unspec -> NaN
# - XGB decides what to do with NaN
#
# (D) For remaining unordered cat vars:
# - Create cat for none_unspec
# - One-hot encode cats
# - Distribute NaN into cats according to their size
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
