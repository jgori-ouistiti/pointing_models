import polars, pandas
import numpy
import matplotlib.pyplot as plt
import json

plt.style.use(["fivethirtyeight"])

path = "segmented_data_JG_060323.json"

with open(path, "r") as _file:
    data = json.load(_file)

rows = []


def deflate_json_mt_id(json_data):
    json_data = data
    for P in json_data.keys():
        Pdata = json_data[P]
        for _file in Pdata.keys():
            _filedata = Pdata[_file]
            D = _filedata.pop("D")
            W = _filedata.pop("W")
            for _mov in _filedata.keys():
                if ("-" in _mov) or (
                    "mov" not in _mov
                ):  # not sure why pop does not work
                    continue  # only movements to the same direction
                _movdata = _filedata[_mov]
                MT = _movdata["t"][-1] - _movdata["t"][0]
                Ae = abs(_movdata["x"][-1] - _movdata["x"][0])
                x = _movdata["x"][-1]
                rows.append([P, D, W, Ae, x, MT])

    return rows


rows = deflate_json_mt_id(data)
df = pandas.DataFrame(rows, columns=["Participant", "A", "W", "Ae", "x", "MT"])
df["ID"] = numpy.log2(1 + df["A"] / df["W"])

print("initial datasetsize")
print(len(df))


rows = []
k = 0
for (P, D, W), group in df.groupby(["Participant", "A", "W"]):
    X = numpy.array(group["x"][1:])
    stdX = (X - numpy.mean(X)) / numpy.std(X)
    keep = numpy.where(numpy.abs(stdX) < 2)[0]
    sigma_x = numpy.std(X[keep])
    we = 4.133 * sigma_x
    IDe = numpy.log2(1 + numpy.mean(group["Ae"]) / we)
    ID = numpy.log2(1 + D / W)
    for n, (_, row) in enumerate(group.iterrows()):
        if n in keep + 1:
            # continue
            rows.append([P, k, ID, IDe, row[5], D, W])

    k += 1

df = pandas.DataFrame(rows, columns=["participant", "set", "ID", "IDe", "MT", "d", "w"])

df["D"] = df["d"] * 36 / 1000
df["W"] = df["w"] * 36 / 100

print("datasetsize after first outlier removal")
print(len(df))

df = df[df["IDe"] >= 1]
print("datasetsize after second outlier removal")
print(len(df))
df.to_csv("exchange/pdd/all.csv")


import seaborn


fig1, ax = plt.subplots(1, 1)
seaborn.scatterplot(data=df, x="ID", y="IDe", ax=ax, s=200, alpha=0.5)

plt.ion()
plt.tight_layout()
plt.show()


xlim = ax.get_xlim()
ax.plot(xlim, xlim, "-", label="IDe = ID")
import statsmodels.api as sm
import statsmodels.formula.api as smf

md = smf.mixedlm(
    formula="IDe~ ID*W + ID*D",
    data=df,
    groups=df["participant"],
)
mdf = md.fit(reml=False, method=["Powell", "L-BFGS-B"])

print("============FULL==========")
print(mdf.summary())

md = smf.mixedlm(
    formula="IDe~ ID*W",
    data=df,
    groups=df["participant"],
)
mdf = md.fit(reml=False, method=["Powell", "L-BFGS-B"])
print("============W==========")
print(mdf.summary())

md = smf.mixedlm(
    formula="IDe~ ID*D",
    data=df,
    groups=df["participant"],
)
mdf = md.fit(reml=False, method=["Powell", "L-BFGS-B"])
print("============D==========")
print(mdf.summary())

md = smf.mixedlm(
    formula="IDe~ID",
    data=df,
    groups=df["participant"],
)
mdf = md.fit(reml=False, method=["Powell", "L-BFGS-B"])
print("============SHORT==========")
print(mdf.summary())


plt.ion()
plt.tight_layout()
# fig1.savefig("img/id_ide_yamanaka.pdf")
plt.show()
exit()


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
