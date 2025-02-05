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


df = pandas.read_csv("../R/exchange/loglik_part_df.csv")

df["nk"] = numpy.where(numpy.isnan(df["params2"]), 1, 2)
df.loc[df["copula"] == "Independent", "nk"] = 0
df = df[~numpy.isinf(df["ll"])]

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


fig, axs = plt.subplots(2, 2)

seaborn.heatmap(
    ll_array,
    annot=True,
    linewidth=0.5,
    xticklabels=copulas_labels,
    yticklabels=[f"P{i}" for i in range(15)],
    cmap=cmap,
    ax=axs[0, 0],
)
seaborn.heatmap(
    row_normalized_ll_array,
    annot=True,
    linewidth=0.5,
    xticklabels=copulas_labels,
    yticklabels=[f"P{i}" for i in range(15)],
    cmap=cmap,
    ax=axs[0, 1],
)
seaborn.heatmap(
    AICnormalized_array,
    annot=True,
    linewidth=0.5,
    xticklabels=copulas_labels,
    yticklabels=[f"P{i}" for i in range(15)],
    cmap=cmap,
    ax=axs[1, 0],
)
seaborn.heatmap(
    model_evidence_ratio,
    annot=True,
    linewidth=0.5,
    xticklabels=copulas_labels,
    yticklabels=[f"P{i}" for i in range(15)],
    cmap=cmap,
    ax=axs[1, 1],
)

for ax in axs.ravel():
    ax.set_xticklabels(copulas_labels, rotation=45)


fig, ax = plt.subplots(1, 1)
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
# fig.savefig("img/model_evidence_ratio_jgp_mer.pdf")

plt.show()
