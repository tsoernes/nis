import numpy as np
import pandas as pd
import sklearn as skl
import xgboost as xgb  # noqa
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split

from load import load_prep

# df = pd.read_pickle("trainvalid-oh.pkl")
df = load_prep(use_codes=True)
target = 'SalePrice'
train_df, test_df = train_test_split(df, test_size=0.1)
train_y, test_y = train_df.pop(target), test_df.pop(target)
log_test_y = np.log(test_y)


def print_losses(pred_y):
    r2 = r2_score(y_true=test_y, y_pred=pred_y)
    mse = mean_squared_error(y_true=test_y, y_pred=pred_y)
    rmse = np.sqrt(mse)
    log_pred_y = np.log(pred_y)
    log_r2 = r2_score(y_true=log_test_y, y_pred=log_pred_y)
    log_mse = mean_squared_error(y_true=log_test_y, y_pred=log_pred_y)
    log_rmse = np.sqrt(log_mse)
    print(f"Loss/log-loss:\nR2:{r2:.4f}/{log_r2:.4f}\n"
          f"MSE:{mse:.4f}/{log_mse:.4f}\nRMSE:{rmse:.4f}/{log_rmse:.4f}")


def gridsearch():
    gs_params = {
        'scoring': 'neg_mean_squared_log_error',
        'n_jobs': -1,
        # log(pred+1) can give error if pred=-1, in which case return -inf (ish)
        'error_score': -3.4e+37,
        'cv': 5
    }
    # Grid search max tree depth and min child weight
    # Best = 12, 3
    bounds = {'max_depth': [9, 11, 12], 'min_child_weight': [3, 6, 10]}
    optimized_GBM = GridSearchCV(
        estimator=xgb.XGBRegressor(missing=0), param_grid=bounds, **gs_params)
    optimized_GBM.fit(train_df, train_y)
    print(optimized_GBM.grid_scores_)

    bounds2 = {'learning_rate': [0.2, 0.1], 'subsample': [0.9]}
    optimized_GBM = GridSearchCV(
        # xgb.XGBRegressor(missing=0, **optimized_GBM.best_params_),
        xgb.XGBRegressor(missing=0, max_depth=12, min_child_weight=3),
        param_grid=bounds2,
        **gs_params)
    optimized_GBM.fit(train_df, train_y)
    print(optimized_GBM.grid_scores_)


reg = xgb.XGBRegressor(
    n_estimators=200, missing=0, max_depth=12, min_child_weight=3, eta=0.1, n_jobs=-1)
reg = reg.fit(train_df, train_y)
pred_y = reg.predict(test_df)
print_losses(pred_y)
