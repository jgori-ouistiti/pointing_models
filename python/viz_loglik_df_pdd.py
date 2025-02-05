import seaborn
import pandas
import matplotlib.pyplot as plt
import numpy

copulas_labels = [
    "Independent",
    "Gaussian",
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


df = pandas.read_csv("../R/exchange/loglik_df_pdd_part.csv")

df["nk"] = numpy.where(numpy.isnan(df["params2"]), 1, 2)
df.loc[df["copula"] == "Independent", "nk"] = 0
df = df[~numpy.isinf(df["ll"])]
df = df[~numpy.isnan(df["ll"])]

import numpy
from matplotlib.colors import LinearSegmentedColormap, LogNorm

cmap = LinearSegmentedColormap.from_list("rg", ["r", "g"], N=256)

participants = df["P"].unique()
copulas = df["copula"].unique()
ll_array = numpy.empty((len(participants), len(copulas)))
AIC_array = numpy.empty((len(participants), len(copulas)))
for np, p in enumerate(participants):
    df_part = df[(df["P"] == p)]
    for nc, c in enumerate(copulas):
        df_cop = df_part[(df_part["copula"] == c)]
        ll_array[np, nc] = df_cop["ll"].sum()
        # AIC_array[np, nc] = (
        #     2 * df_cop["nk"].unique() * len(df_cop["ll"]) - 2 * df_cop["ll"].sum()
        # )
        AIC_array[np, nc] = 2 * df_cop["nk"].unique() - 2 * df_cop["ll"].mean()


row_normalized_ll_array = ll_array / numpy.max(ll_array, axis=1).reshape(-1, 1)
AICnormalized_array = numpy.maximum(
    -20, ll_array - numpy.max(ll_array, axis=1).reshape(-1, 1)
)

model_evidence_ratio = numpy.exp(
    (numpy.min(AIC_array, axis=1).reshape(-1, 1) - AIC_array) / 2
)


fig, ax = plt.subplots(1, 1)
seaborn.heatmap(
    numpy.maximum(1e-4, model_evidence_ratio),
    annot=True,
    fmt=".1e",
    linewidth=0.5,
    xticklabels=copulas_labels,
    yticklabels=[f"P{i}" for i in range(12)],
    cmap=cmap,
    ax=ax,
    norm=LogNorm(),
)
ax.set_xticklabels(copulas_labels, rotation=45)
plt.ion()
plt.tight_layout()
# fig.savefig("img/model_evidence_ratio_pdd.pdf")
plt.show()

rho = df[df["copula"] == "tCopula"]["params1"].mean()
nu = df[df["copula"] == "tCopula"]["params2"]

theta = df[df["copula"] == "rot gumbelCopula"]["params1"].mean()
