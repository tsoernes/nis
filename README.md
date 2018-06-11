```
p3 run.py
Loss/log-loss:
R2:0.9121/0.9090
MSE:47928668.7609/0.0441
RMSE:6923.0534/0.2100
```


How to handle features with missing values:
(excluding imputing missing value with e.g. mean or using nearest neighbors)

(none_unspec abbreviates 'None or Unspecified'
category counts exclude NaN and none_unspec)

(A) For variables (vars) with 1 category (cat):
- Any catvar must have 2 cats. Create cat N for none_unspec
- Order cats (X, N)
- XGB decides what to do with NaN

(B) For vars with 2 cats (X < Y) and no none_unspec:
- Order cats (X, Y)
- XGB decides what to do with NaN

(C) For vars with 2 cats and none_unspec or >2 cats:
- Create cat N for none_unspec
- One-Hot encode cats
- Propagate NaNs to OH cats

(D) For vars with >2 ordered cats:
- Convert none_unspec -> NaN
- XGB decides what to do with NaN

(E) For continuous vars:
- Convert none_unspec, NaN -> 0
- XGB decides what to do with 0


XGB decides at training time whether missing values go into the right or left node.
It chooses which to minimise loss.

P.S:
The above procedure was not performed for the reported result, due to insufficient RAM
to store data-set in memory.

Further possible improvements:
- Use large unordered categories as-is (even if incorrect); convert small cats to one-hot
- Parse horsepower or torque from `fiProductClassDesc`
- Use month and day of week (i.e. 'Monday') from 'saledate', not just year
