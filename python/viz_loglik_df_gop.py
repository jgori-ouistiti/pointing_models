import seaborn
import pandas
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

cmap = LinearSegmentedColormap.from_list("rg", ["r", "g"], N=256)

label_mapping = {
    "1": "Speed Emphasis",
    "2": "Speed",
    "3": "Balanced",
    "4": "Accuracy",
    "5": "Accuracy Emphasis",
}

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

df = pandas.read_csv("../R/exchange/loglik_df_gop.csv")
df.drop(df.columns[[0, -1]], axis=1, inplace=True)
df.index = copulas_labels


cm = seaborn.light_palette("green", as_cmap=True)
df.style.background_gradient(cmap=cm)
df_styled = df.style.background_gradient(cmap=cm).format("{:.0f}")

import numpy

df = pandas.read_csv("../R/exchange/loglik_part_gop.csv")
df = df[~numpy.isnan(df["ll"])]

# data reads I but is actually strategy, see GOP_analysis_csv_version.py
strategies = df["I"].unique()
copulas = df["copula"].unique()
ll_array = numpy.empty((len(strategies), len(copulas)))

for ns, s in enumerate(strategies):
    df_s = df[(df["I"] == s)]
    for nc, c in enumerate(copulas):
        df_cop = df_s[(df_s["copula"] == c)]
        ll_array[ns, nc] = df_cop["ll"].mean()


row_normalized_ll_array = ll_array / numpy.max(ll_array, axis=1).reshape(-1, 1)

fig, ax = plt.subplots(1, 1)
seaborn.heatmap(row_normalized_ll_array, annot=True, linewidth=0.5, cmap=cmap)
ax.set_xticklabels(copulas_labels, rotation=45)
ax.set_yticklabels(list(label_mapping.values()), rotation=45)
plt.tight_layout()
plt.ion()

fig.savefig("img/ll_gop_per_strat.pdf")


df = pandas.read_csv("../R/exchange/loglik_part_gop_no_iteration.csv")

participants = df["P"].unique()
copulas = df["copula"].unique()
ll_array = numpy.empty((len(participants), len(copulas)))

for np, p in enumerate(participants):
    df_part = df[(df["P"] == p)]
    for nc, c in enumerate(copulas):
        df_cop = df_part[(df_part["copula"] == c)]
        ll_array[np, nc] = df_cop["ll"]

row_normalized_ll_array = ll_array / numpy.max(ll_array, axis=1).reshape(-1, 1)
fig, ax = plt.subplots(1, 1)
seaborn.heatmap(
    row_normalized_ll_array,
    annot=True,
    linewidth=0.5,
    cmap=cmap,
)
ax.set_xticklabels(copulas_labels, rotation=45)
ax.set_yticklabels([f"P{p}" for p in range(15)], rotation=0)

plt.ion()
plt.tight_layout()
fig.savefig("img/ll_gop_per_P.pdf")
