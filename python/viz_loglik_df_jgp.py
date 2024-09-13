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

import numpy
from matplotlib.colors import LinearSegmentedColormap

cmap = LinearSegmentedColormap.from_list("rg", ["r", "g"], N=256)

participants = df["P"].unique()
copulas = df["copula"].unique()
ll_array = numpy.empty((len(participants), len(copulas)))

for np, p in enumerate(participants):
    df_part = df[(df["P"] == p)]
    for nc, c in enumerate(copulas):
        df_cop = df_part[(df_part["copula"] == c)]
        ll_array[np, nc] = df_cop["ll"].mean()

row_normalized_ll_array = ll_array / numpy.max(ll_array, axis=1).reshape(-1, 1)

fig, ax = plt.subplots(1, 1)
seaborn.heatmap(
    row_normalized_ll_array,
    annot=True,
    linewidth=0.5,
    xticklabels=copulas_labels,
    yticklabels=[f"P{i}" for i in range(15)],
    cmap=cmap,
)
ax.set_xticklabels(copulas_labels, rotation=45)
plt.ion()
plt.tight_layout()
fig.savefig("img/ll_jgp.pdf")

plt.show()
