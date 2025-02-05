import seaborn
import pandas
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import numpy

plt.style.use(["fivethirtyeight"])

cmap = LinearSegmentedColormap.from_list("rg", ["r", "g"], N=256)

label_mapping = {
    1: "Speed Emphasis",
    2: "Speed",
    3: "Balanced",
    4: "Accuracy",
    5: "Accuracy Emphasis",
}

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

cmap = LinearSegmentedColormap.from_list("rg", ["r", "g"], N=256)


import numpy

df = pandas.read_csv("../R/exchange/loglik_go_part_strat.csv")
df["nk"] = numpy.where(numpy.isnan(df["params2"]), 1, 2)
df.loc[df["copula"] == "Independent", "nk"] = 0
df = df[~numpy.isinf(df["ll"])]

# for cop in df["copula"].unique():
#     print(cop, len(df[df["copula"] == cop]))
# Independent 75
# normalCopula 75
# claytonCopula 74
# gumbelCopula 75
# galambosCopula 54
# huslerReissCopula 54
# tevCopula 52
# tCopula 69
# rot gumbelCopula 75
# rot galambosCopula 54
# rot huslerReissCopula 54


strategies = df["S"].unique()
copulas = df["copula"].unique()
participants = df["P"].unique()


ll_array = numpy.empty((len(strategies), len(copulas)))
AIC_array = numpy.empty((len(strategies), len(copulas)))
N_array = numpy.empty((len(strategies), len(copulas)))
for ns, s in enumerate(strategies):
    df_s = df[(df["S"] == s)]
    for nc, c in enumerate(copulas):
        df_cop = df_s[(df_s["copula"] == c)]
        ll_array[ns, nc] = df_cop["ll"].sum()
        AIC_array[ns, nc] = 2 * df_cop["nk"].unique() * len(
            df_cop["ll"]
        ) - 2 * numpy.sum(df_cop["ll"])
        N_array[ns, nc] = len(df_cop["ll"])

fig1, ax = plt.subplots(1, 1)
seaborn.heatmap(
    N_array,
    annot=True,
    linewidth=0.5,
    xticklabels=copulas_labels,
    yticklabels=label_mapping.values(),
    cmap=cmap,
    ax=ax,
)
ax.set_xticklabels(copulas_labels, rotation=45)
plt.ion()
plt.tight_layout()
# fig1.savefig("img/number_of_successful_copula_fits_go_strat.pdf")
plt.show()

# copulas = [
#     "Independent",
#     "normalCopula",
#     "claytonCopula",
#     "gumbelCopula",
#     "tCopula",
#     "rot gumbelCopula",
# ]
# copulas_labels = [
#     "Independent",
#     "Gaussian",
#     "Clayton",
#     "Gumbel",
#     "t",
#     "rotGumbel",
# ]

ll_array = numpy.empty((len(strategies), len(copulas)))
AIC_array = numpy.empty((len(strategies), len(copulas)))
for ns, s in enumerate(strategies):
    df_s = df[(df["S"] == s)]
    for nc, c in enumerate(copulas):
        df_cop = df_s[(df_s["copula"] == c)]
        ll_array[ns, nc] = df_cop["ll"].sum()
        AIC_array[ns, nc] = 2 * df_cop["nk"].unique() * len(
            df_cop["ll"]
        ) - 2 * numpy.sum(df_cop["ll"])


model_evidence_ratio = numpy.exp(
    (numpy.min(AIC_array, axis=1).reshape(-1, 1) - AIC_array) / 2
)

fig2, ax = plt.subplots(1, 1)
seaborn.heatmap(
    numpy.maximum(1e-4, model_evidence_ratio),
    annot=True,
    fmt=".1e",
    linewidth=0.5,
    xticklabels=copulas_labels,
    yticklabels=label_mapping.values(),
    cmap=cmap,
    ax=ax,
    norm=LogNorm(),
)
ax.set_xticklabels(copulas_labels, rotation=45)
plt.ion()
plt.tight_layout()
plt.show()
# fig2.savefig("img/ll_gop_per_strat.pdf")
print(df["S"].unique())
df_balanced = df[(df["S"] == 2) & (df["copula"] == "galambosCopula")]
fig20, ax = plt.subplots(1, 1)
seaborn.violinplot(data=df_balanced, x="params1", ax=ax)
seaborn.stripplot(
    data=df_balanced, x="params1", color="black", size=5, jitter=True, ax=ax
)
ax.set_xlabel(r"$\theta$")
plt.tight_layout()
plt.ion()
plt.show()
# fig20.savefig("img/galambos_params.pdf")

df_gauss = df[df["copula"] == "normalCopula"]
from scipy.stats import ttest_1samp

results = []
for group, group_data in df_gauss.groupby("S"):
    t_stat, p_value = ttest_1samp(group_data["params1"], 0)
    results.append(
        {
            "Group": group,
            "T-Statistic": t_stat,
            "P-Value": p_value,
            "Mean": group_data["params1"].mean(),
            "Count": len(group_data),
        }
    )

df = pandas.read_csv("../R/exchange/loglik_go_part.csv")
df["nk"] = numpy.where(numpy.isnan(df["params2"]), 1, 2)
df.loc[df["copula"] == "Independent", "nk"] = 0
df = df[~numpy.isinf(df["ll"])]

participants = df["P"].unique()
copulas = df["copula"].unique()
ll_array = numpy.empty((len(participants), len(copulas)))
AIC_array = numpy.empty((len(participants), len(copulas)))
N_array = numpy.empty((len(participants), len(copulas)))

for np, p in enumerate(participants):
    df_part = df[(df["P"] == p)]
    for nc, c in enumerate(copulas):
        df_cop = df_part[(df_part["copula"] == c)]
        ll_array[np, nc] = df_cop["ll"].sum()
        AIC_array[np, nc] = 2 * df_cop["nk"].unique() * len(
            df_cop["ll"]
        ) - 2 * numpy.sum(df_cop["ll"])
        N_array[np, nc] = len(df_cop["ll"])

model_evidence_ratio = numpy.exp(
    (numpy.min(AIC_array, axis=1).reshape(-1, 1) - AIC_array) / 2
)


fig3, ax = plt.subplots(1, 1)
seaborn.heatmap(
    numpy.maximum(1e-4, model_evidence_ratio),
    annot=True,
    fmt=".1e",
    linewidth=0.5,
    xticklabels=copulas_labels,
    yticklabels=[f"P{i}" for i in range(15)],
    cmap=cmap,
    ax=ax,
    norm=LogNorm(),
)
ax.set_xticklabels(copulas_labels, rotation=45)
plt.ion()
plt.tight_layout()
plt.show()
# fig3.savefig("img/ll_gop_per_P.pdf")

# df_galambos = df[df["copula"] == "galambosCopula"]
# sum = 0
# N = 0
# for p in [0, 3, 9, 12]:
#     df_tmp = df_galambos[df_galambos["P"] == p]
#     sum += df_tmp["params1"].sum()
#     N += len(df_tmp)
# print(sum / N)

# fig4, axs = plt.subplots(2, 2)

# for nc, cop in enumerate(["normalCopula", "tCopula"]):
#     df_cop = df[df["copula"] == cop]
#     seaborn.barplot(data=df_cop, y="params1", x="P", errorbar=("ci", 95), ax=axs[0, nc])
#     seaborn.barplot(data=df_cop, y="params2", x="P", errorbar=("ci", 95), ax=axs[1, nc])
#     axs[0, nc].set_ylabel(cop + "  params1")

# axs[1,1].set_sycale('log')
# plt.ion()
# plt.tight_layout()
# plt.show()
# fig4.savefig("img/params_gop_P.pdf")

fig5, axs = plt.subplots(1, 2)
df_cop = df[df["copula"] == "tCopula"]
seaborn.barplot(data=df_cop, y="params1", x="P", errorbar=("ci", 95), ax=axs[0])
seaborn.barplot(data=df_cop, y="params2", x="P", errorbar=("ci", 95), ax=axs[1])
axs[0].set_ylabel(r"$\rho$")
axs[1].set_ylabel(r"$\nu$")
axs[1].set_yscale("log")
plt.tight_layout()
plt.ion()
plt.show()
# fig5.savefig("img/t_cop_parameters.pdf")
print(df_cop["params1"].mean())
