import polars
import numpy
import matplotlib.pyplot as plt

plt.style.use(["fivethirtyeight"])

path = "/home/juliengori/Documents/VC/gen_pointing_stat/python/ToJulien_CHI2023Yamanaka/Exp1BubbleCursor/allData.csv"

df = polars.read_csv(path)
pointing_data = df.filter(polars.col("isBubble") == False)
pointing_data = pointing_data.with_columns(
    (numpy.log2(1 + polars.col("A") / polars.col("W"))).alias("ID")
)
participants = pointing_data["participantNumber"].unique()
strategies = pointing_data["bias"].unique()
Ds = pointing_data["A"].unique()
Ws = pointing_data["W"].unique()
WeW = pointing_data["EW/W"].unique()

t0 = pointing_data[0]["targetX"]

rows = []
k = 0
for _d in Ds:
    for _w in Ws:
        for p in participants:
            for s in strategies:
                for nw, _wew in enumerate(WeW):
                    p_data = pointing_data.filter(
                        (polars.col("participantNumber") == p)
                        & (polars.col("A") == _d)
                        & (polars.col("W") == _w)
                        & (polars.col("bias") == s)
                        & (polars.col("EW/W") == _wew)
                    )

                    # we = (
                    #     4.133
                    #     * numpy.sqrt(
                    #         (p_data["x"] - p_data["targetX"]) ** 2
                    #         + (p_data["y"] - p_data["targetY"]) ** 2
                    #     )
                    #     .sqrt()
                    #     .sum()
                    #     / (len(p_data))
                    # )
                    A = p_data["A"].unique().item()
                    W = p_data["W"].unique().item()
                    # IDe = numpy.log2(1 + A / we)
                    ID = p_data["ID"].unique().item()

                    initial_position = (0, 0)

                    X = []
                    Y = []

                    for n, row in enumerate(p_data.iter_rows()):
                        x0, y0 = initial_position
                        tx = row[12]
                        ty = row[13]
                        x = row[8]
                        y = row[9]
                        alpha = numpy.arctan2(ty - y0, tx - x0)
                        x, y = x - x0, y - y0
                        R = numpy.array(
                            [
                                [numpy.cos(alpha), numpy.sin(alpha)],
                                [-numpy.sin(alpha), numpy.cos(alpha)],
                            ]
                        )
                        v = numpy.array([x, y])
                        x, y = R @ v
                        # if (k==123) and (nw ==2):
                        #     print(nw, n,x,y)
                        X.append(x)
                        Y.append(y)
                        initial_position = (row[12], row[13])
                    # if (k==123) and (nw == 0):
                    #     exit()

                    X = numpy.array(X[1:])
                    stdX = (X - numpy.mean(X)) / numpy.std(X)
                    keep = numpy.where(numpy.abs(stdX) < 2.5)[0]
                    sigma_x = numpy.std(X[keep])
                    we = 4.133 * sigma_x
                    IDe = numpy.log2(1 + A / we)
                    for n, row in enumerate(p_data.iter_rows()):
                        if n in keep + 1:
                            # continue
                            rows.append([p, k, s, ID, IDe, row[11], A, W])

                k += 1


p_data = pointing_data.filter(
    (polars.col("participantNumber") == 5)
    & (polars.col("A") == 770)
    & (polars.col("W") == 8)
    & (polars.col("bias") == 0)
)

sets = p_data["set"].unique()
fig, axs = plt.subplots(1, 3)
for ns, s in enumerate(sets):
    sp_data = p_data.filter((polars.col("set") == s))
    axs[ns].plot(sp_data["targetX"], sp_data["targetY"], "*")
    axs[ns].plot(sp_data["x"], sp_data["y"], "o")

plt.ion()
plt.show()
# exit()

df = polars.DataFrame(
    rows, schema=["participant", "set", "strategy", "ID", "IDe", "MT", "A", "W"]
)
print(len(df))
exit()
df_mean = df.group_by("set").mean()
df_mean = df_mean.with_columns(
    polars.col(["set", "participant", "strategy"]).cast(polars.Int16)
)
df_mean = df_mean.with_columns(
    polars.col("ID").round(3)
)  # floating point precision meant 9 levels for ID instead of 6


import seaborn
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal


# from GOP_analysis_csv_version.py
def custom_fit(df, strategy, ax, **kwargs):

    fc = kwargs.pop("fc", "None")
    facecolor = kwargs.pop("facecolor", fc)
    ec = kwargs.pop("ec", "r")
    edgecolor = kwargs.pop("edgecolor", ec)
    lw = kwargs.pop("lw", 2)

    df1 = df.filter(polars.col("strategy") == strategy)
    df2 = df1.select(["IDe", "MT"])
    mean, cov = multivariate_normal.fit(df2)
    eigenvalues, eigenvectors = numpy.linalg.eigh(cov)
    angle = numpy.degrees(numpy.arctan2(*eigenvectors[:, 0][::-1]))
    width, height = 2 * numpy.sqrt(5.991 * eigenvalues)

    ellipse = Ellipse(
        xy=mean,
        width=width,
        height=height,
        angle=angle,
        edgecolor=edgecolor,
        facecolor=facecolor,
        lw=lw,
        **kwargs,
    )
    ax.add_patch(ellipse)
    ax.plot(df1["IDe"], df1["MT"], "*")
    return mean, cov, ax, angle, width, height


colors = ["#1B3668", "#D42A2F", "#2D7A3D"]
labels = ["Speed", "Balanced", "Accuracy"]

rows = []
fig, ax = plt.subplots(1, 1)
seaborn.scatterplot(data=df_mean, x="IDe", y="MT", ax=ax)
for ns, s in enumerate([0, 1, 2]):
    for n, _id in enumerate(df_mean["ID"].unique()):
        filtered_df = df_mean.filter(polars.col("ID") == _id)
        if n == 0:
            mean, cov, _, _, _, _ = custom_fit(
                filtered_df, s, ax, ec=colors[ns], label=labels[ns]
            )
        else:
            mean, cov, _, _, _, _ = custom_fit(filtered_df, s, ax, ec=colors[ns])
        rows.append(
            [
                *mean,
                cov[0, 0],
                cov[1, 1],
                cov[0, 1] / numpy.sqrt(cov[0, 0] * cov[1, 1]),
                _id,
                s,
            ]
        )
plt.legend()
plt.tight_layout()
plt.ion()
# fig.savefig('img/yamanaka_ide_mmt.pdf')
plt.show()

df = polars.DataFrame(
    rows, schema=["mu_i", "mu_mt", "sigma_i", "sigma_t", "rho", "ID", "s"]
)
_map = {0: -0.5, 1: 0, 2: 0.5}
_map_str = {0: "Speed", 1: "Balanced", 2: "Accuracy"}
df = df.with_columns(
    polars.col("s").cast(polars.Float64).replace(_map).alias("strategy_num")
)
df = df.with_columns(
    polars.col("s").cast(polars.String).replace(_map_str).alias("strategy")
)

_xlims = [df["ID"].min(), df["ID"].max()]
fontdict = {"fontsize": 14}

fig2, axs = plt.subplots(1, 5)
import statsmodels.formula.api as smf

wmd = smf.ols(formula="mu_i~ID ", data=df)  # AIC = 7
wmd = smf.ols(formula="mu_i~strategy_num ", data=df)  # AIC = 59
wmd = smf.ols(formula="mu_i~1 ", data=df)  # AIC = 58
wmd = smf.ols(formula="mu_i~ID + strategy_num", data=df)  # AIC = -30

result = wmd.fit()
a, b, c = list(result.params)
for ns, s in enumerate([-0.5, 0, 0.5]):
    axs[0].plot(_xlims, [a + b * _x + c * s for _x in _xlims], "-", label=labels[ns])
seaborn.scatterplot(data=df, x="ID", y="mu_i", hue="strategy", ax=axs[0], s=100)
axs[0].set_title(
    r"$\mu_i$ = "
    + f"{a:.2f} + {b:.2f} ID + {c:.2f} s, "
    + r"$r^2$"
    + f"={result.rsquared:.2f}",
    fontdict=fontdict,
)
axs[0].set_ylabel(r"$\mu_i$")

wmd = smf.ols(formula="mu_mt~ID ", data=df)  # AIC = -14
wmd = smf.ols(formula="mu_mt~strategy_num ", data=df)  # AIC = 17
wmd = smf.ols(formula="mu_mt~1 ", data=df)  # AIC = 17
wmd = smf.ols(formula="mu_mt~ID + strategy_num", data=df)  # AIC = -34


result = wmd.fit()
a, b, c = list(result.params)
for ns, s in enumerate([-0.5, 0, 0.5]):
    axs[1].plot(_xlims, [a + b * _x + c * s for _x in _xlims], "-", label=labels[ns])
seaborn.scatterplot(data=df, x="ID", y="mu_mt", hue="strategy", ax=axs[1], s=100)
axs[1].set_title(
    r"$\mu_t$ = "
    + f"{a:.2f} + {b:.2f} ID + {c:.2f} s, "
    + r"$r^2$"
    + f"={result.rsquared:.2f}",
    fontdict=fontdict,
)
axs[1].set_ylabel(r"$\mu_t$")
rho_f_df = df.filter(polars.col("rho") > 0.2)


wmd = smf.ols(formula="sigma_i~ID + strategy_num", data=df)  # -39.52 AIC
wmd = smf.ols(formula="sigma_i~strategy_num", data=df)  # -40.36 AIC
wmd = smf.ols(formula="sigma_i~ID", data=df)  # -40.36 AIC

wmd = smf.ols(formula="sigma_i~1", data=df)  # -38.94 AIC

result = wmd.fit()
a = list(result.params)[0]
axs[2].plot(_xlims, [a for _x in _xlims], "-")
axs[2].set_title(r"$\sigma_i$ = " + f"{a:.2f}", fontdict=fontdict)
axs[2].set_ylabel(r"$\sigma_i$")
seaborn.scatterplot(data=df, x="ID", y="sigma_i", hue="strategy", ax=axs[2], s=100)

wmd = smf.ols(formula="sigma_t~ID + strategy_num", data=df)  # -107 AIC
wmd = smf.ols(formula="sigma_t~strategy_num", data=df)  # -90.26 AIC
wmd = smf.ols(formula="sigma_t~1", data=df)  # -91 AIC
wmd = smf.ols(formula="sigma_t~ID", data=df)  # -106 AIC
result = wmd.fit()


a, b = list(result.params)
axs[3].plot(_xlims, [a + b * _x for _x in _xlims], "-")
axs[3].set_title(
    r"$\sigma_t$ = " + f"{a:.2f} + {b:.2f} ID " + r"$r^2$" + f"={result.rsquared:.2f}",
    fontdict=fontdict,
)
axs[3].set_ylabel(r"$\sigma_t$")
seaborn.scatterplot(data=df, x="ID", y="sigma_t", hue="strategy", ax=axs[3], s=100)


wmd = smf.ols(formula="rho~ID + strategy_num", data=df)  # AIC = -6
wmd = smf.ols(formula="rho~ID ", data=df)  # AIC = -7.5
wmd = smf.ols(formula="rho~strategy_num", data=df)  # AIC = -4
wmd = smf.ols(formula="rho~1", data=df)  # AIC = -5.7

result = wmd.fit()
a = list(result.params)[0]
axs[4].plot(_xlims, [a for _x in _xlims], "-")
axs[4].set_title(r"$\rho$ = " + f"{a:.2f}", fontdict=fontdict)
axs[4].set_ylabel(r"$\rho$")
seaborn.scatterplot(data=df, x="ID", y="rho", hue="strategy", ax=axs[4], s=100)

plt.legend()
plt.ion()
# fig2.savefig(fname="img/yamanaka_biv_strategy_fit.pdf")
plt.show()
