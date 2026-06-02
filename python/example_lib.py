import matplotlib.pyplot as plt
import seaborn
import polars
import statsmodels.formula.api as smf

data = "./exchange/yamanaka/all.csv"

df = polars.read_csv(data).to_pandas()

# df_part = df[df["participant"] == 0]

fig, axs = plt.subplots(1, 2)
seaborn.scatterplot(data=df, x="IDe", y="MT", ax=axs[0])
md = smf.mixedlm(formula="MT~IDe*strategy_num", data=df, groups=df["participant"])
plt.ion()
plt.show()
