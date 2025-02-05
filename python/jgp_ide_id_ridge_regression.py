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
df = df.rename(columns={"IDe(2d)": "IDe"})
df = df[["IDe", "ID", "A", "W", "participant"]]


df = df[~df["IDe"].duplicated(keep="first")]
# df = df / df.std()

X = df[["ID", "A", "W", "participant"]]
X.describe(include="all")

y = df["IDe"].ravel()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=999)

from sklearn.linear_model import RidgeCV

alphas = numpy.logspace(-5, 5, 21)
ridge_cv = RidgeCV(alphas=alphas, cv=None, scoring="neg_mean_squared_error")
ridge_cv.fit(X_train, y_train)
print(f"Best alpha: {ridge_cv.alpha_}")

# Predictions
y_pred = ridge_cv.predict(X_test)

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

best_alpha = ridge_cv.alpha_
ridge_model = Ridge(alpha=best_alpha)
ridge_model.fit(X_train, y_train)
y_pred_ridge = ridge_model.predict(X_test)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_ridge, color="blue", alpha=0.7, label="Predicted vs Actual")
plt.plot(
    [min(y_test), max(y_test)],
    [min(y_test), max(y_test)],
    color="red",
    linestyle="--",
    label="Perfect Prediction",
)

# Add labels and legend
plt.title("Ridge Regression: Predicted vs Actual Values")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()


exit()

import statsmodels.api as sm
import statsmodels.formula.api as smf

fig, axs = plt.subplots(1, 2)
seaborn.scatterplot(data=df, x="ID", y="ide", hue="W", ax=axs[0])
seaborn.scatterplot(data=df, x="ID", y="ide", hue="A", ax=axs[1])


df["D"] = df["A"] / 1000
df["w"] = df["W"] / 1000

# scaling for better fits

id_vec = numpy.array([df["ID"].min(), df["ID"].max()])


md = smf.mixedlm(formula="ide~ID*w*D", data=df, groups=df["participant"])
mdf = md.fit()
print(mdf.summary().as_latex())


wmd = smf.mixedlm(formula="ide~ID*w", data=df, groups=df["participant"])
mdf = wmd.fit()
a, b, c, d, _ = mdf.params
W = sorted(df["W"].unique())

for w in W:
    w = w / 1000
    axs[0].plot(
        id_vec, a + b * id_vec + c * w + d * (w * id_vec), "-", label=f"{w*1000}"
    )
axs[0].legend()
print(mdf.summary().as_latex())


wmd = smf.mixedlm(formula="ide~ID*D", data=df, groups=df["participant"])
mdf = wmd.fit()
a, b, c, d, _ = mdf.params
D = sorted(df["A"].unique())
for _d in D:
    _d = _d / 1000
    axs[1].plot(
        id_vec, a + b * id_vec + c * _d + d * (_d * id_vec), "-", label=f"{_d*1000}"
    )
axs[1].legend()
print(mdf.summary().as_latex())

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
