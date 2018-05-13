import pandas as pd
import sklearn as skl
import xgboost as xgb  # noqa
from sklearn.model_selection import GridSearchCV, train_test_split

# from load import load_prep

# df = load_prep(oh=True)
df = pd.read_pickle("trainvalid-oh.pkl")
target = 'SalePrice'
train_df, test_df = train_test_split(df, test_size=0.1)
train_y, test_y = train_df.pop(target), test_df.pop(target)

# regressor = dfml.xgboost.XGBRegressor()
# train_df.fit(regressor)
# predicted = test_df.predict(regressor)

# Grid search max tree depth and min child weight
ind_params = {
    'learning_rate': 0.1,
    'n_estimators': 100,
    'seed': 0,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'reg:linear'
}
cv_params = {'max_depth': [5, 7, 9], 'min_child_weight': [1, 3, 5]}
optimized_GBM = GridSearchCV(
    xgb.XGBRegressor(missing=0), cv_params, scoring='neg_mean_squared_error', n_jobs=-1)
optimized_GBM.fit(train_df, train_y)
print(optimized_GBM.grid_scores_)

# xgdmat = xgb.DMatrix(final_train,
#                      y_train)  # Create our DMatrix to make XGBoost more efficient
# our_params = {
#     'eta': 0.1,
#     'seed': 0,
#     'subsample': 0.8,
#     'colsample_bytree': 0.8,
#     'objective': 'binary:logistic',
#     'max_depth': 3,
#     'min_child_weight': 1
# }
# # Grid Search CV optimized settings

# cv_xgb = xgb.cv(
#     params=our_params,
#     dtrain=xgdmat,
#     num_boost_round=3000,
#     nfold=5,
#     metrics=['error'
#              ],  # Make sure you enter metrics inside a list or you may encounter issues!
#     early_stopping_rounds=100)  # Look for early stopping that minimizes error
# cv_xgb.tail(5)
# xgb.plot_importance(final_gb)
