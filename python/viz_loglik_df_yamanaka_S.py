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


df = pandas.read_csv("../R/exchange/loglik_yamanaka_s.csv")

df["nk"] = numpy.where(numpy.isnan(df["params2"]), 1, 2)
df.loc[df["copula"] == "Independent", "nk"] = 0
df = df[~numpy.isinf(df["ll"])]
# for cop in df["copula"].unique():
#     print(cop, len(df[df["copula"] == cop]))
# Independent 36
# normalCopula 36
# claytonCopula 36
# gumbelCopula 36
# galambosCopula 36
# huslerReissCopula 36
# tevCopula 36
# tCopula 36
# rot gumbelCopula 36
# rot galambosCopula 36
# rot huslerReissCopula 36


import numpy
from matplotlib.colors import LinearSegmentedColormap, LogNorm

cmap = LinearSegmentedColormap.from_list("rg", ["r", "g"], N=256)

S = df["S"].unique()
P = df["P"].unique()
copulas = df["copula"].unique()
ll_array = numpy.empty((len(S) * len(P), len(copulas)))
AIC_array = numpy.empty((len(S) * len(P), len(copulas)))

Slabels = ["Speed", "Balanced", "Precise"]

nn = 0
ylabels = []
for ns, s in enumerate(S):
    for np, p in enumerate(P):
        df_cond = df[(df["S"] == s) & (df["P"] == p)]
        for nc, c in enumerate(copulas):
            df_cop = df_cond[(df_cond["copula"] == c)]
            if len(df_cop) == 0:
                continue
            ll_array[nn, nc] = df_cop["ll"].sum()
            AIC_array[nn, nc] = 2 * df_cop["nk"].unique() * len(
                df_cop["ll"]
            ) - 2 * numpy.sum(df_cop["ll"])
        ylabels.append(f"{Slabels[s]}xP{p}")
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
plt.show()
# fig1.savefig("img/R_yamanaka_SP.pdf")


df_gauss = df[df["copula"] == "normalCopula"]

fig2, ax = plt.subplots(1, 1)
seaborn.barplot(data=df_gauss, x="S", y="params1", ax=ax)
ax.set_ylabel(r"$\rho$")
# fig2.savefig("img/rho_SP_yamanaka.pdf")
import statsmodels.api as sm
import statsmodels.formula.api as smf

numpy.mean(df_gauss["params1"])
wmd = smf.mixedlm(formula="params1 ~ S", data=df_gauss, groups=df_gauss["P"])
mdf = wmd.fit()
print(mdf.summary().as_latex())

df_t = df[df["copula"] == "tCopula"]
df_t["lognu"] = numpy.log(df_t["params2"])
_map = {0: -0.5, 1: 0, 2: 0.5}
df_t["strategy_num"] = df_t["S"].replace(to_replace=_map).astype("float")
_map = {0: "Speed", 1: "Balanced", 2: "Accuracy"}
df_t["strategy"] = df_t["S"].replace(to_replace=_map)

fig3, axs = plt.subplots(1, 2)
seaborn.barplot(data=df_t, x="strategy", y="params1", ax=axs[0])
axs[0].set_ylabel(r"$\rho$")
seaborn.barplot(data=df_t, x="strategy", y="lognu", ax=axs[1])
axs[1].set_ylabel(r"$\log(\nu)$")

# fig3.savefig("img/rho_nu_SP_yamanaka.pdf")
import statsmodels.api as sm
import statsmodels.formula.api as smf


numpy.mean(df_t["params1"])
wmd = smf.mixedlm(formula="params1 ~ strategy_num", data=df_t, groups=df_t["P"])
mdf = wmd.fit()
print(mdf.summary().as_latex())

numpy.mean(df_t["params2"])
wmd = smf.mixedlm(formula="lognu ~ strategy_num", data=df_t, groups=df_t["P"])
mdf = wmd.fit()
print(mdf.summary().as_latex())

plt.show()
