import pandas as pd
from matplotlib import pyplot as plt

from load import load_prep

# Prepped
df_p = load_prep(oh=False, use_codes=False)

# Raw
df_r = pd.read_pickle('trainvalid-wdates.pkl')

# Saledate has year only
price_year_ax = df_p.groupby("saledate")['SalePrice'].mean().plot()
plt.show()

print("\nGrouser type distribution")
print(df_r['Grouser_Type'].value_counts(dropna=False))

print(
    "\nNumber of categories (including NaN and None) for each variable recognizes as a catvar"
)
cat_cols = filter(lambda x: df_p[x].dtype.name == 'category', df_p.columns)
for x in cat_cols:
    print(x, len(df_p[x].unique()), df_p[x].unique())
