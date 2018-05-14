```
p3 run.py
Loss/log-loss:
R2:0.9121/0.9090
MSE:47928668.7609/0.0441
RMSE:6923.0534/0.2100
```

Some statistics:
```
In [1889]: df['MachineHoursCurrentMeter'].max()
Out[1889]: 2483300.0
```
Mean machine hours
```
In [1910]: df['MachineHoursCurrentMeter'][df['MachineHoursCurrentMeter'] > 0].mean()
Out[1910]: 7053.819247997828
```
Stick length data distribution
```
In [1946]: df['Stick_Length'].value_counts(dropna=False)
Out[1949]:
NaN                    310437
None or Unspecified     81539
9' 6"                    5832
10' 6"                   3519
11' 0"                   1601
9' 10"                   1463
9' 8"                    1462
9' 7"                    1423
12' 10"                  1087
...
Name: Stick_Length, dtype: int64
```

How to handle features with missing values:
(excluding imputing missing value with e.g. mean or using nearest neighbors)

none_unspec - 'None or Unspecified'
cat counts excludes NaN and none_unspec

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
Note that the gblinear booster treats missing values as zeros.

P.S:
The above procedure was not performed for the reported result, due to insufficient RAM
to store data-set in memory.

Further possible improvements:
- Use large unordered categories as-is (even if incorrect); convert small cats to one-hot
- Parse horsepower or torque from `fiProductClassDesc`
- Use month and day of week (i.e. 'Monday') from 'saledate', not just year
