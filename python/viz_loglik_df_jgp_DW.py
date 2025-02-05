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

# df = pandas.read_csv("../R/exchange/loglik_df_jgp.csv")
# df.drop(df.columns[[0]], axis=1, inplace=True)
# df.index = copulas_labels

# columns = [f"P{p}_i{i}" for p in range(1, 16) for i in range(1, 7)]
# df.columns = columns

# cm = seaborn.light_palette("green", as_cmap=True)
# df.style.background_gradient(cmap=cm)
# df_styled = df.style.background_gradient(cmap=cm).format("{:.0f}")


# with open("dump/table.html", "w") as _file:
#     _file.write(df_styled.to_html())


df = pandas.read_csv("../R/exchange/loglik_df_jgp_D_W.csv")

df["nk"] = numpy.where(numpy.isnan(df["params2"]), 1, 2)
df.loc[df["copula"] == "Independent", "nk"] = 0
df = df[~numpy.isinf(df["ll"])]

# >>> for cop in df["copula"].unique():
# ...     print(cop, len(df[df["copula"] == cop]))
# ...
# Independent 12
# normalCopula 12
# claytonCopula 12
# gumbelCopula 12
# tCopula 12
# rot gumbelCopula 12
# galambosCopula 4
# huslerReissCopula 4
# tevCopula 2
# rot galambosCopula 4
# rot huslerReissCopula 4


copulas = [
    "Independent",
    "normalCopula",
    "claytonCopula",
    "gumbelCopula",
    "tCopula",
    "rot gumbelCopula",
]
copulas_labels = [
    "Independent",
    "Gaussian",
    "Clayton",
    "Gumbel",
    "t",
    "rotGumbel",
]


import numpy
from matplotlib.colors import LinearSegmentedColormap, LogNorm

cmap = LinearSegmentedColormap.from_list("rg", ["r", "g"], N=256)

D = df["D"].unique()
W = df["W"].unique()
# copulas = df["copula"].unique()
ll_array = numpy.empty((len(D) * len(W), len(copulas)))
AIC_array = numpy.empty((len(D) * len(W), len(copulas)))


nn = 0
ylabels = []
for nd, d in enumerate(D):
    for nw, w in enumerate(W):
        print(
            d,
            w,
        )
        df_cond = df[(df["D"] == d) & (df["W"] == w)]
        for nc, c in enumerate(copulas):
            df_cop = df_cond[(df_cond["copula"] == c)]
            if len(df_cop) == 0:
                continue
            ll_array[nn, nc] = df_cop["ll"]
            AIC_array[nn, nc] = 2 * df_cop["nk"].unique() - 2 * df_cop["ll"]
        ylabels.append(f"{d}x{w}")
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
# fig.savefig("img/R_DW.pdf")
plt.show()


df_64_gauss = df[(df["copula"] == "normalCopula") & (df["W"] == 64)]
