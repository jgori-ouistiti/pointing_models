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
                    keep = numpy.where(numpy.abs(stdX) < 2)[0]
                    sigma_x = numpy.std(X[keep])
                    we = 4.133 * sigma_x
                    IDe = numpy.log2(1 + A / we)
                    for n, row in enumerate(p_data.iter_rows()):
                        if n in keep + 1:
                            # continue
                            rows.append([p, k, s, ID, IDe, row[11], A, W])

                k += 1

df = polars.DataFrame(
    rows, schema=["participant", "set", "strategy", "ID", "IDe", "MT", "A", "W"]
)
df.write_csv("exchange/yamanaka/all.csv")

import seaborn

custom_labels = {
    "0": "2: Speed",
    "1": "3: Balanced",
    "2": "4: Accuracy",
}


df = df.to_pandas()
import copy

df_all = copy.copy(df)
df = df[~df["IDe"].duplicated(keep="first")]

_map = {0: -0.5, 1: 0, 2: 0.5}
df["strategy_num"] = df["strategy"].replace(_map).astype("float")
df["D"] = df["A"] / 1000
df["w"] = df["W"] / 100

fig1, ax = plt.subplots(1, 1)
seaborn.scatterplot(data=df, x="ID", y="IDe", hue="strategy", ax=ax, s=200, alpha=0.5)


xlim = ax.get_xlim()
ax.plot(xlim, xlim, "-", label="IDe = ID")
import statsmodels.api as sm
import statsmodels.formula.api as smf

md = smf.mixedlm(
    formula="IDe~strategy_num + ID*w + ID*D",
    data=df,
    groups=df["participant"],
)
mdf = md.fit(reml=True, method=["Powell", "L-BFGS-B"])

print("============FULL==========")
print(mdf.summary())

md = smf.mixedlm(
    formula="IDe~strategy_num*w+ ID*w",
    data=df,
    groups=df["participant"],
)
mdf = md.fit(reml=True, method=["Powell", "L-BFGS-B"])
print("============W==========")
print(mdf.summary())

md = smf.mixedlm(
    formula="IDe~strategy_num*D+ ID*D",
    data=df,
    groups=df["participant"],
)
mdf = md.fit(reml=True, method=["Powell", "L-BFGS-B"])
print("============D==========")
print(mdf.summary())

md = smf.mixedlm(
    formula="IDe~ID*strategy_num",
    data=df,
    groups=df["participant"],
)
mdf = md.fit(reml=True, method=["Powell", "L-BFGS-B"])
print("============SHORT==========")
print(mdf.summary())

a, b, c, d, _ = mdf.params

for ns, s in enumerate(iterable=[-0.5, 0, 0.5]):
    ax.plot(
        xlim,
        [a + b * _x + c * s + d * s * _x for _x in xlim],
        "-",
        label=f"{custom_labels[str(ns)]}",
    )

md = smf.mixedlm(
    formula="IDe~ID + strategy_num",
    data=df,
    groups=df["participant"],
)
mdf = md.fit(reml=False, method=["Powell", "L-BFGS-B"])
print(mdf.summary())

ax.legend()
for text in ax.legend().get_texts()[:3]:
    text.set_text(custom_labels[text._text])
plt.ion()
plt.tight_layout()
# fig1.savefig("img/id_ide_yamanaka.pdf")
plt.show()
fig2, axs = plt.subplots(1, 2)


xlim = [df["ID"].min(), df["ID"].max()]
seaborn.scatterplot(
    data=df_all, x="ID", y="MT", hue="strategy", alpha=0.7, ax=axs[0], s=30
)
md = smf.mixedlm(formula="MT~ID*strategy_num", data=df, groups=df["participant"])
mdf = md.fit()
print(mdf.summary().as_latex())
a, b, c, d, _ = mdf.params
for ns, s in enumerate(iterable=[-0.5, 0, 0.5]):
    axs[0].plot(
        xlim,
        [a + b * _x + c * s + d * s * _x for _x in xlim],
        "-",
        label=f"{custom_labels[str(ns)]}",
    )
axs[0].legend()
for text in axs[0].legend().get_texts()[:3]:
    text.set_text(custom_labels[text._text])

seaborn.scatterplot(
    data=df_all, x="IDe", y="MT", hue="strategy", alpha=0.7, ax=axs[1], s=30
)
md = smf.mixedlm(formula="MT~IDe*strategy_num", data=df, groups=df["participant"])
mdf = md.fit()
print(mdf.summary().as_latex())

xlim = [df["IDe"].min(), df["IDe"].max()]

a, b, c, d, _ = mdf.params
for ns, s in enumerate(iterable=[-0.5, 0, 0.5]):
    axs[1].plot(
        xlim,
        [a + b * _x + c * s + d * s * _x for _x in xlim],
        "-",
        label=f"{custom_labels[str(ns)]}",
    )
axs[1].legend()
for text in axs[1].legend().get_texts()[:3]:
    text.set_text(custom_labels[text._text])
plt.ion()
plt.tight_layout()
# fig2.savefig("img/fitts_yamanaka.pdf")
plt.show()
