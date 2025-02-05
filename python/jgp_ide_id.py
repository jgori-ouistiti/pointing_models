import pandas
import numpy
import matplotlib.pyplot as plt
import seaborn
import warnings

warnings.filterwarnings("ignore")

import emg_arbitrary_variance as emg_av


plt.style.use("fivethirtyeight")

df = pandas.read_csv(filepath_or_buffer="JGP_per_xp.csv")


## plots make sense, now fit with EMG, and try to have variance evolve with strategy

params = {"beta0": [], "beta1": [], "sigma": [], "expo0": [], "expo1": []}
fontdict = {"fontsize": 14}

df = df[df["iteration"] <= 3]
df = df.rename(columns={"IDe(2d)": "ide"})


df = df[~df["ide"].duplicated(keep="first")]


import statsmodels.api as sm
import statsmodels.formula.api as smf

fig, axs = plt.subplots(1, 2)
seaborn.scatterplot(data=df, x="ID", y="ide", hue="W", ax=axs[0])
seaborn.scatterplot(data=df, x="ID", y="ide", hue="A", ax=axs[1])


df["D"] = df["A"] / 1000
df["w"] = df["W"] / 100

# scaling for better fits

id_vec = numpy.array([df["ID"].min(), df["ID"].max()])


md = smf.mixedlm(formula="ide~ID*w*D", data=df, groups=df["participant"])
mdf = md.fit(reml=False)
print("======FULL")
print(mdf.summary().as_latex())


wmd = smf.mixedlm(formula="ide~ID*w", data=df, groups=df["participant"])
mdf = wmd.fit(reml=False)
print("======W")
print(mdf.summary().as_latex())
a, b, c, d, _ = mdf.params
W = sorted(df["W"].unique())

for w in W:
    w = w / 1000
    axs[0].plot(
        id_vec, a + b * id_vec + c * w + d * (w * id_vec), "-", label=f"{w*1000}"
    )
axs[0].legend()


wmd = smf.mixedlm(formula="ide~ID*D", data=df, groups=df["participant"])
mdf = wmd.fit(reml=False)
print("======D")
print(mdf.summary().as_latex())
a, b, c, d, _ = mdf.params
D = sorted(df["A"].unique())
for _d in D:
    _d = _d / 1000
    axs[1].plot(
        id_vec, a + b * id_vec + c * _d + d * (_d * id_vec), "-", label=f"{_d*1000}"
    )
axs[1].legend()


wmd = smf.mixedlm(formula="ide~ID", data=df, groups=df["participant"])
mdf = wmd.fit(reml=True, method=["Powell", "L-BFGS-B"])
print("======ID")
print(mdf.summary().as_latex())


wmd = smf.ols(formula="ide~ID", data=df, groups=df["participant"])
mdf = wmd.fit()
print(mdf.summary().as_latex())
exit()


handles, labels = axs[0].get_legend_handles_labels()
axs[0].legend(handles, ["W=" + label for label in labels])
handles, labels = axs[1].get_legend_handles_labels()
axs[1].legend(handles, ["D=" + label for label in labels])
axs[0].set_ylabel(r"ID$_e$")
axs[1].set_ylabel(r"ID$_e$")


plt.ion()
plt.tight_layout()
# plt.savefig('img/ide_id_jpg.pdf')
plt.show()
