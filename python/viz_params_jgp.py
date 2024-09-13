import seaborn
import pandas
import matplotlib.pyplot as plt

copulas_labels = [
    "Clayton",
    "Gumbel",
    "Galambos",
    "HR",
    "t-EV",
    "t",
    "rotGumbel",
    "rotGalambos",
    "rotHR",
]


df = pandas.read_csv("../R/exchange/loglik_part_df.csv")

import numpy
from matplotlib.colors import LinearSegmentedColormap


fig, axs = plt.subplots(1, 2)

df_tcop = df[df["copula"] == "tCopula"]
seaborn.barplot(data=df_tcop, y="params", x="P", errorbar=("ci", 95), ax=axs[0])
df_rgcop = df[df["copula"] == "rot gumbelCopula"]
seaborn.barplot(data=df_rgcop, y="params", x="P", errorbar=("ci", 95), ax=axs[1])


plt.ion()
plt.tight_layout()
fig.savefig("img/params_jgp.pdf")

plt.show()
