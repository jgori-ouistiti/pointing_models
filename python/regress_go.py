import pandas
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn

plt.style.use(["fivethirtyeight"])

df = pandas.read_csv(filepath_or_buffer="GO_df.csv")

mean_mts = df.groupby("IDe").mean().reset_index()

md = smf.ols(formula="MT~IDe", data=mean_mts).fit()

fig, ax = plt.subplots(1, 1)
# seaborn.scatterplot(
#     data=mean_mts, x="IDe", y="MT", hue=df["strategy"].astype("category"), ax=ax
# )
seaborn.scatterplot(data=mean_mts, x="IDe", y="MT", hue="strategy", ax=ax)


xlims = mean_mts["IDe"].min(), mean_mts["IDe"].max()
a, b = md.params
ax.plot(
    xlims,
    [a + b * _x for _x in xlims],
    "-",
    label=r"$r^2(\text{ID}_e,\text{MT})=$" + f"{md.rsquared:.2f}",
)


grouped_repet = mean_mts.groupby("strategy").mean().reset_index()
ax.plot(
    grouped_repet["IDe"],
    grouped_repet["MT"],
    "D",
    ms=10,
    label="aggregates per strategy",
)
md = smf.ols(formula="MT~IDe", data=grouped_repet).fit()
a, b = md.params
ax.plot(
    xlims,
    [a + b * _x for _x in xlims],
    "-",
    label=r"$r^2(\text{ID}_e,\text{MT})=$" + f"{md.rsquared:.2f}",
)

handles, labels = ax.get_legend_handles_labels()
labels = [
    "Speed Emphasis",
    "Speed",
    "Balanced",
    "Accuracy",
    "Accuracy Emphasis",
] + labels[5:]
ax.legend(handles, labels)
ax.set_xlabel(r"ID$_e$ (bit)")
ax.set_ylabel(r"$\overline{\text{MT}}$" + "(s)")
# fig.savefig("img/fittslaw_guiard.pdf")
plt.ion()
plt.tight_layout()
plt.show()
