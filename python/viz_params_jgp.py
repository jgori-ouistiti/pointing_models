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


fig, axs = plt.subplots(2, 11)

for nc, cop in enumerate(df["copula"].unique()):
    df_cop = df[df["copula"] == cop]
    seaborn.barplot(data=df_cop, y="params1", x="P", errorbar=("ci", 95), ax=axs[0, nc])
    seaborn.barplot(data=df_cop, y="params2", x="P", errorbar=("ci", 95), ax=axs[1, nc])
    axs[0, nc].set_ylabel(cop + "  params1")


plt.ion()
plt.tight_layout()
fig.savefig("img/params_jgp.pdf")

plt.show()
