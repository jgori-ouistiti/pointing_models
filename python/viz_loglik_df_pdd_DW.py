import seaborn

import numpy
import pandas
import matplotlib.pyplot as plt

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


df = pandas.read_csv("../R/exchange/loglik_df_pdd_D_W.csv")


df["nk"] = numpy.where(numpy.isnan(df["params2"]), 1, 2)
df.loc[df["copula"] == "Independent", "nk"] = 0
df = df[~numpy.isinf(df["ll"])]
df = df[~numpy.isnan(df["ll"])]

# for cop in df["copula"].unique():
#     print(cop, len(df[df["copula"] == cop]))

# Independent 8
# normalCopula 8
# claytonCopula 8
# gumbelCopula 8
# tCopula 8
# rot gumbelCopula 8
# galambosCopula 6
# huslerReissCopula 6
# tevCopula 6
# rot galambosCopula 6
# rot huslerReissCopula 6


import numpy
from matplotlib.colors import LinearSegmentedColormap, LogNorm

cmap = LinearSegmentedColormap.from_list("rg", ["r", "g"], N=256)

W = df["W"].unique()
copulas = df["copula"].unique()

Wlabels = ["{:g}".format(float("{:.2g}".format(i))) for i in W]
# copulas = df["copula"].unique()
ll_array = numpy.empty((len(W), len(copulas)))
AIC_array = numpy.empty((len(W), len(copulas)))


nn = 0
ylabels = []
for nw, w in enumerate(W):
    df_cond = df[df["W"] == w]
    for nc, c in enumerate(copulas):
        df_cop = df_cond[(df_cond["copula"] == c)]
        if len(df_cop) == 0:
            continue
        ll_array[nn, nc] = df_cop["ll"]
        AIC_array[nn, nc] = 2 * df_cop["nk"].unique() - 2 * df_cop["ll"]
    ylabels.append(f"{Wlabels[nw]}")
    nn += 1

model_evidence_ratio = numpy.exp(
    (numpy.min(AIC_array, axis=1).reshape(-1, 1) - AIC_array) / 2
)

fig, ax = plt.subplots(1, 1)
seaborn.heatmap(
    numpy.maximum(1e-2, model_evidence_ratio),
    annot=True,
    fmt=".1e",
    linewidth=0.5,
    xticklabels=copulas_labels,
    yticklabels=ylabels,
    cmap=cmap,
    ax=ax,
    norm=LogNorm(),
)
ax.set_xticklabels(copulas_labels, rotation=45)
plt.ion()
plt.tight_layout()
# fig.savefig("img/R_DW_pdd.pdf")
plt.show()

theta = numpy.asarray(df[df["copula"] == "galambosCopula"]["params1"])
W = numpy.asarray(df[df["copula"] == "galambosCopula"]["W"])

plt.style.use(["fivethirtyeight"])
fig, ax = plt.subplots(1, 1)
ax.plot(W, theta, "-")
ax.set_xlabel("W")
ax.set_ylabel(r"$\theta$")
plt.ion()
plt.tight_layout()
fig.savefig("img/theta_w_pdd.pdf")
