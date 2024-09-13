import pickle
import seaborn
import matplotlib.pyplot as plt
import pingouin
import statsmodels.api as sm


plt.style.use("fivethirtyeight")

with open("gop_association_agg_strategy.pkl", "rb") as _file:
    df = pickle.load(_file)

df = df.to_pandas()

r_icc = pingouin.intraclass_corr(
    data=df, raters="repetition", targets="Participant", ratings="r", nan_policy="omit"
)
rho_icc = pingouin.intraclass_corr(
    data=df,
    raters="repetition",
    targets="Participant",
    ratings="rho",
    nan_policy="omit",
)
tau_icc = pingouin.intraclass_corr(
    data=df,
    raters="repetition",
    targets="Participant",
    ratings="tau",
    nan_policy="omit",
)

with open(".tmp.txt", "w") as _file:
    _file.write(r_icc.to_latex())
    _file.write(rho_icc.to_latex())
    _file.write(tau_icc.to_latex())


fig, ax = plt.subplots(1, ncols=1)
seaborn.histplot(df, x="r", ax=ax, label="Pearson r")
seaborn.histplot(df, x="rho", ax=ax, label=r"Spearman $\rho$")
seaborn.histplot(df, x="tau", ax=ax, label=r"Kendall $\tau$")
ax.legend()
ax.set_xlabel("Association measure")
plt.tight_layout()
plt.ion()
fig.savefig("supp_source/association_gop_agg_strategy.pdf")
plt.show()

with open("gop_association_agg_repetition.pkl", "rb") as _file:
    df = pickle.load(_file)

df = df.to_pandas()

r_icc = pingouin.intraclass_corr(
    data=df, raters="strategy", targets="Participant", ratings="r", nan_policy="omit"
)
rho_icc = pingouin.intraclass_corr(
    data=df,
    raters="strategy",
    targets="Participant",
    ratings="rho",
    nan_policy="omit",
)
tau_icc = pingouin.intraclass_corr(
    data=df,
    raters="strategy",
    targets="Participant",
    ratings="tau",
    nan_policy="omit",
)

with open(".tmp.txt", "a") as _file:
    _file.write(r_icc.to_latex())
    _file.write(rho_icc.to_latex())
    _file.write(tau_icc.to_latex())


grouped_mean = df[["r", "rho", "tau", "strategy"]].groupby("strategy").mean()

with open(".tmp.txt", "a") as _file:
    _file.write(grouped_mean.to_latex())


fig, axs = plt.subplots(1, ncols=3)
# seaborn.histplot(df, x="r", hue="strategy", ax=axs[0], label="Pearson r")
# seaborn.histplot(df, x="rho", hue="strategy", ax=axs[1], label=r"Spearman $\rho$")
# seaborn.histplot(df, x="tau", hue="strategy", ax=axs[2], label=r"Kendall $\tau$")
strategy = df["strategy"]
X = sm.add_constant(strategy)
Y = df["r"]
model = sm.OLS(Y, X)
res = model.fit()


for ax in axs:
    ax.legend()
axs[1].set_xlabel("Association measure")
plt.tight_layout()
plt.ion()
plt.show()
