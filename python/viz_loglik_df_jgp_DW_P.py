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


df = pandas.read_csv("../R/exchange/loglik_df_jgp_D_W_P.csv")

df["nk"] = numpy.where(numpy.isnan(df["params2"]), 1, 2)
df.loc[df["copula"] == "Independent", "nk"] = 0
df = df[~numpy.isinf(df["ll"])]


# >>> for cop in df['copula'].unique():
# ...     print(cop, len(df[df['copula'] == cop]))
# ...
# Independent 180
# normalCopula 180
# claytonCopula 180
# gumbelCopula 180
# tCopula 170
# rot gumbelCopula 180
# galambosCopula 78
# huslerReissCopula 78
# tevCopula 86
# rot galambosCopula 78
# rot huslerReissCopula 78

for cop in df["copula"].unique():
    print(cop, len(df[df["copula"] == cop]))


import numpy
from matplotlib.colors import LinearSegmentedColormap, LogNorm

cmap = LinearSegmentedColormap.from_list("rg", ["r", "g"], N=256)

D = df["D"].unique()
W = df["W"].unique()
P = df["P"].unique()
copulas = df["copula"].unique()
ll_array = numpy.empty((len(D) * len(W) * len(P), len(copulas)))
AIC_array = numpy.empty((len(D) * len(W) * len(P), len(copulas)))


nn = 0
ylabels = []
for nd, d in enumerate(D):
    for nw, w in enumerate(W):
        for np, p in enumerate(P):
            print(d, w, p)
            df_cond = df[(df["D"] == d) & (df["W"] == w) & (df["P"] == p)]
            for nc, c in enumerate(copulas):
                df_cop = df_cond[(df_cond["copula"] == c)]
                if ~len(df_cop):
                    continue
                ll_array[nn, nc] = df_cop["ll"].sum()
                AIC_array[nn, nc] = 2 * df_cop["nk"].unique() * len(
                    df_cop["ll"]
                ) - 2 * numpy.sum(df_cop["ll"])
                ylabels.append(f"{d}x{w}, P{p}")
            nn += 1

model_evidence_ratio = numpy.exp(
    (numpy.min(AIC_array, axis=1).reshape(-1, 1) - AIC_array) / 2
)

# fig, ax = plt.subplots(1, 1)
# seaborn.heatmap(
#     numpy.maximum(1e-3, model_evidence_ratio),
#     annot=True,
#     fmt=".1e",
#     linewidth=0.5,
#     xticklabels=copulas_labels,
#     # yticklabels=ylabels,
#     cmap=cmap,
#     ax=ax,
#     norm=LogNorm(),
# )
# ax.set_xticklabels(copulas_labels, rotation=45)
# plt.ion()
# plt.tight_layout()


### notice that Galambos, HR, t-EV rotGalambos and rotHR do not fit at all, with many ll = -inf. remove those before summing on participants

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


# copulas_labels = ["Independent", "Gaussian", "Clayton", "Gumbel", "t", "rotGumbel"]
ll_array = numpy.empty((len(D) * len(W), len(copulas)))
AIC_array = numpy.empty((len(D) * len(W), len(copulas)))
N_array = numpy.empty((len(D) * len(W), len(copulas)))


nn = 0
ylabels = []
for nd, d in enumerate(D):
    for nw, w in enumerate(W):
        # for np, p in enumerate(P):
        print(d, w)
        ylabels.append(f"{d}x{w}")
        df_cond = df[(df["D"] == d) & (df["W"] == w)]  # & (df["P"] == p)]
        for nc, c in enumerate(copulas):
            print(c, len(df_cop["ll"]))
            df_cop = df_cond[(df_cond["copula"] == c)]
            df_cop = df_cop[~numpy.isinf(df_cop["ll"])]
            ll_array[nn, nc] = numpy.sum(df_cop["ll"])
            AIC_array[nn, nc] = 2 * df_cop["nk"].unique() * len(
                df_cop["ll"]
            ) - 2 * numpy.sum(df_cop["ll"])
            N_array[nn, nc] = len(df_cop["ll"])
        nn += 1


row_normalized_ll_array = ll_array / numpy.max(ll_array, axis=1).reshape(-1, 1)
AICnormalized_array = numpy.maximum(
    -20, ll_array - numpy.max(ll_array, axis=1).reshape(-1, 1)
)

model_evidence_ratio = numpy.exp(
    (numpy.min(AIC_array, axis=1).reshape(-1, 1) - AIC_array) / 2
)

fig2, ax = plt.subplots(1, 1)
seaborn.heatmap(
    N_array,
    annot=True,
    linewidth=0.5,
    xticklabels=copulas_labels,
    yticklabels=ylabels,
    cmap=cmap,
    ax=ax,
)
ax.set_xticklabels(copulas_labels, rotation=45)
plt.ion()
plt.tight_layout()
# fig2.savefig("img/number_of_successful_copula_fits_jgp_dw.pdf")
plt.show()

copulas = [
    "Independent",
    "normalCopula",
    "claytonCopula",
    "gumbelCopula",
    "galambosCopula",
    "huslerReissCopula",
]
copulas_labels = [
    "Independent",
    "Gaussian",
    "Clayton",
    "Gumbel",
    "Galambos",
    "HR",
]

ll_array = numpy.empty((len(D) * len(W), len(copulas)))
AIC_array = numpy.empty((len(D) * len(W), len(copulas)))


nn = 0
ylabels = []
for nd, d in enumerate(D):
    for nw, w in enumerate(W):
        # for np, p in enumerate(P):
        print(d, w)
        ylabels.append(f"{d}x{w}")
        df_cond = df[(df["D"] == d) & (df["W"] == w)]  # & (df["P"] == p)]
        for nc, c in enumerate(copulas):
            df_cop = df_cond[(df_cond["copula"] == c)]
            df_cop = df_cop[~numpy.isinf(df_cop["ll"])]
            ll_array[nn, nc] = numpy.sum(df_cop["ll"])
            AIC_array[nn, nc] = 2 * df_cop["nk"].unique() * len(
                df_cop["ll"]
            ) - 2 * numpy.sum(df_cop["ll"])
        nn += 1


row_normalized_ll_array = ll_array / numpy.max(ll_array, axis=1).reshape(-1, 1)
AICnormalized_array = numpy.maximum(
    -20, ll_array - numpy.max(ll_array, axis=1).reshape(-1, 1)
)

model_evidence_ratio = numpy.exp(
    (numpy.min(AIC_array, axis=1).reshape(-1, 1) - AIC_array) / 2
)


fig3, ax = plt.subplots(1, 1)
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
# fig3.savefig("img/model_evidence_ratio_DW_jgp.pdf")

plt.show()
exit()


df_t = df[df["copula"] == "tCopula"]
df_t["ID"] = numpy.log2(1 + df_t["D"] / df_t["W"])


fig, axs = plt.subplots(3, 2)
seaborn.scatterplot(data=df_t, x="W", y="params1", ax=axs[0, 0], hue="P")
seaborn.scatterplot(data=df_t, x="W", y="params2", ax=axs[0, 1], hue="P")
seaborn.scatterplot(data=df_t, x="D", y="params1", ax=axs[1, 0], hue="P")
seaborn.scatterplot(data=df_t, x="D", y="params2", ax=axs[1, 1], hue="P")
seaborn.scatterplot(data=df_t, x="ID", y="params1", ax=axs[2, 0], hue="P")
seaborn.scatterplot(data=df_t, x="ID", y="params2", ax=axs[2, 1], hue="P")

import statsmodels.api as sm
import statsmodels.formula.api as smf

df_t = df_t.dropna()


def print_file_tex(path, result):
    with open(path, "w") as _file:
        _file.write(result.summary().as_latex())


wmd = smf.mixedlm(formula="params1 ~ W", data=df_t, groups=df_t["P"])
mdf = wmd.fit()
print_file_tex("__res_W_rho.txt", mdf)
a, b, sigma_sq = mdf.params
for p, v in mdf.random_effects.items():
    axs[0, 0].plot(W, [a + b * w + v for w in W], "-", label=f"P{p}")

wmd = smf.mixedlm(formula="params2 ~ W", data=df_t, groups=df_t["P"])
mdf = wmd.fit()
print_file_tex("__res_W_nu.txt", mdf)
a, b, sigma_sq = mdf.params
for p, v in mdf.random_effects.items():
    axs[0, 1].plot(W, [a + b * w + v for w in W], "-", label=f"P{p}")

dmd = smf.mixedlm(formula="params1 ~ D", data=df_t, groups=df_t["P"])
mdf = dmd.fit()
print_file_tex("__res_D_rho.txt", mdf)
a, b, sigma_sq = mdf.params
for p, v in mdf.random_effects.items():
    axs[1, 0].plot(D, [a + b * d + v for d in D], "-", label=f"P{p}")

dmd = smf.mixedlm(formula="params2 ~ D", data=df_t, groups=df_t["P"])
mdf = dmd.fit()
print_file_tex("__res_D_nu.txt", mdf)
a, b, sigma_sq = mdf.params
for p, v in mdf.random_effects.items():
    axs[1, 1].plot(D, [a + b * d + v for d in D], "-", label=f"P{p}")

id_vec = [numpy.min(df_t["ID"]), numpy.max(df_t["ID"])]

idmd = smf.mixedlm(formula="params1 ~ ID", data=df_t, groups=df_t["P"])
mdf = idmd.fit()
print_file_tex("__res_ID_rho.txt", mdf)
a, b, sigma_sq = mdf.params
for p, v in mdf.random_effects.items():
    axs[2, 0].plot(id_vec, [a + b * _id + v for _id in id_vec], "-", label=f"P{p}")
idmd = smf.mixedlm(formula="params2 ~ ID", data=df_t, groups=df_t["P"])
mdf = idmd.fit()
print_file_tex("__res_ID_nu.txt", mdf)
a, b, sigma_sq = mdf.params
for p, v in mdf.random_effects.items():
    axs[2, 1].plot(id_vec, [a + b * _id + v for _id in id_vec], "-", label=f"P{p}")

plt.show()
