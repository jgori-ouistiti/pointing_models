import pandas
import seaborn
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

plt.style.use("fivethirtyeight")


df = pandas.read_csv("../R/exchange/tcop_part_gop.csv").dropna()
df["Participant"] = df["P"].astype(int).astype(str)
df["strategy"] = df["S"]

fig, axs = plt.subplots(1, 2)
seaborn.scatterplot(df, x="strategy", y="rho1", hue="Participant", ax=axs[0])
seaborn.scatterplot(df, x="strategy", y="nu", hue="Participant", ax=axs[1])
axs[1].set_yscale("log")
axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))
axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))
axs[0].set_ylabel(r"$\rho_1$")
axs[0].set_xlabel("Strategy")
axs[1].set_ylabel(r"$\nu$")
axs[1].set_xlabel("Strategy")
axs[1].set_xlim([-0.5, 6])
axs[0].set_xlim([-0.5, 6])

plt.ion()
plt.tight_layout()
plt.show()
# fig.savefig("supp_source/tcop_values.pdf")

from scipy.stats import gmean

rhos = df[["S", "rho1"]].groupby("S").mean()["rho1"]
nus = df[["S", "nu"]].groupby("S").apply(gmean)

import statsmodels.api as sm
from statsmodels.formula.api import mixedlm

model = mixedlm("rho1 ~ strategy ", df, groups=df["Participant"])
result = model.fit()
print(result.summary().as_latex())
