from itertools import chain

import numpy as np
import pandas as pd
import pandas_ml as pdml
import sklearn as skl
import xgboost as xgb

# from load import load_prep

# df = load_prep(oh=True)
df = pd.load("trainvalid-oh.pkl")
dfml = pdml.ModelFrame(df)
dfml.target_name = 'SalePrice'
assert dfml.has_target()
# train_df, test_df = dfml.model_selection.train_test_split(test_size=0.1)
# train_df, test_df = skl.model_selection.train_test_split(dfml, test_size=0.1)
# Does this work at all?
train_df, test_df = skl.model_selection.train_test_split(
    dfml.data.as_matrix(), dfml.target.as_matrix(), test_size=0.1)

assert train_df.has_target()
regressor = dfml.xgboost.XGBRegressor()

train_df.fit(regressor)

predicted = test_df.predict(regressor)
