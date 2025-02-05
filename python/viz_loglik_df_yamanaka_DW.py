import seaborn

import numpy
import pandas
import matplotlib.pyplot as plt

plt.style.use(["fivethirtyeight"])

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


df = pandas.read_csv("../R/exchange/loglik_yamanaka_DW.csv")

df["nk"] = numpy.where(numpy.isnan(df["params2"]), 1, 2)
df.loc[df["copula"] == "Independent", "nk"] = 0
df = df[~numpy.isinf(df["ll"])]

for cop in df["copula"].unique():
    print(cop, len(df[df["copula"] == cop]))
# # ...
# Independent 72
# normalCopula 72
# claytonCopula 72
# gumbelCopula 72
# galambosCopula 66
# huslerReissCopula 66
# tevCopula 65
# tCopula 72
# rot gumbelCopula 72
# rot galambosCopula 66
# rot huslerReissCopula 66


import numpy
from matplotlib.colors import LinearSegmentedColormap, LogNorm

cmap = LinearSegmentedColormap.from_list("rg", ["r", "g"], N=256)

D = df["D"].unique()
W = df["W"].unique()
copulas = df["copula"].unique()
ll_array = numpy.empty((len(D) * len(W), len(copulas)))
AIC_array = numpy.empty((len(D) * len(W), len(copulas)))

nn = 0
ylabels = []
for nd, d in enumerate(D):
    for nw, w in enumerate(W):
        df_cond = df[(df["D"] == d) & (df["W"] == w)]
        for nc, c in enumerate(copulas):
            df_cop = df_cond[(df_cond["copula"] == c)]
            if len(df_cop) == 0:
                continue
            ll_array[nn, nc] = df_cop["ll"].sum()
            AIC_array[nn, nc] = 2 * df_cop["nk"].unique() * len(
                df_cop["ll"]
            ) - 2 * numpy.sum(df_cop["ll"])
        ylabels.append(f"{d}x{w}")
        nn += 1

model_evidence_ratio = numpy.exp(
    (numpy.min(AIC_array, axis=1).reshape(-1, 1) - AIC_array) / 2
)

fig1, ax = plt.subplots(1, 1)
seaborn.heatmap(
    numpy.maximum(1e-3, model_evidence_ratio),
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
# fig1.savefig("img/R_yamanaka_DW.pdf")


df_gauss = df[df["copula"] == "normalCopula"]

fig2, ax = plt.subplots(1, 1)
seaborn.barplot(data=df_gauss, x="W", y="params1", ax=ax, hue="D")
ax.set_ylabel(r"$\rho$")
# fig2.savefig("img/rho_DW_yamanaka.pdf")
import statsmodels.api as sm
import statsmodels.formula.api as smf

numpy.mean(df_gauss["params1"])
wmd = smf.mixedlm(formula="params1 ~ W*D", data=df_gauss, groups=df_gauss["P"])
mdf = wmd.fit()
print(mdf.summary().as_latex())
