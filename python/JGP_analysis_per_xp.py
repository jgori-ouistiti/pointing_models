import polars
import numpy
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, pearsonr, kendalltau, spearmanr
import statsmodels.api as sm
from matplotlib.patches import Ellipse
import seaborn
import pingouin
import pickle
import pandas
import warnings

warnings.filterwarnings("ignore")

import emg_arbitrary_variance as emg_av


plt.style.use("fivethirtyeight")

df = polars.read_csv("JGP_per_xp.csv", has_header=True, ignore_errors=True)

GEN = False

if GEN:
    df = polars.read_csv("JGP_per_xp.csv", has_header=True, ignore_errors=True)

    ## plots make sense, now fit with EMG, and try to have variance evolve with strategy

    params = {"beta0": [], "beta1": [], "sigma": [], "expo0": [], "expo1": []}
    fontdict = {"fontsize": 14}

    participants = df["participant"].unique()
    # iterations = df["iteration"].unique()
    iterations = [0, 1, 2, 3]  #

    figx, axs = plt.subplots(4, 4)

    parameters = numpy.empty((len(participants), len(iterations), 5))

    association = numpy.empty((len(participants), len(iterations), 3))

    for np, participant in enumerate(participants):
        df_part = df.filter(polars.col("participant") == participant)
        for ni, iteration in enumerate(iterations):

            df_it = df_part.filter(polars.col("iteration") == iteration)

            # rough filtering
            # df_filt = df_it.filter(polars.col("Duration") <= 5)
            # try:

            df_it.write_csv(f"exchange/jgp/{np}_{ni}.csv")

            x = numpy.array(df_it["IDe(2d)"])
            y = numpy.array(df_it["Duration"])

            beta = [0, 0.2]
            sigma = 0.1
            expo_mean = [0.2, 0]

            bounds = [(-1, 1), (0, 1), (1e-5, 1), (0, 1), (0, 5)]

            fit_av = emg_av.fit_emg_arbitrary_variance_model(
                beta, sigma, expo_mean, emg_av.linear_expo_mean, x, y, bounds=bounds
            )
            emg_ll = fit_av.fun

            fig, ax = plt.subplots(1, 1)
            X = sm.add_constant(x)
            ols_fit = sm.OLS(y, X).fit()
            # 2k - 2log(likelihood)
            ols_aic = ols_fit.aic
            emg_aic = 2 * len(fit_av.x) - 2 * emg_ll
            vec = [
                pearsonr(x, y).statistic,
                spearmanr(x, y).statistic,
                kendalltau(x, y).statistic,
            ]
            association[np, ni, :] = vec
            _abs_x = [numpy.min(x), numpy.max(x)]
            ax.plot(x, y, "o")
            ax.plot(x, [fit_av.x[0] + fit_av.x[1] * _x for _x in x], "r-")

            ax.set_title(
                rf"P{participant},I{iteration}: $\mathcal{{R}}$ {numpy.exp(-emg_aic + ols_aic)/2:.1e}. r = {vec[0]:.2f}, $\rho$ = {vec[1]:.2f}, $\tau$ = {vec[2]:.2f}",
                fontdict=fontdict,
            )

            fig.savefig(f"supp_source/emg_{participant}_{iteration}.pdf")
            plt.close(fig)

            parameters[np, ni, :] = fit_av.x

    with open("jgp_association.pkl", "wb") as _file:
        pickle.dump(association, _file)

    with open("jgp_parameters.pkl", "wb") as _file:
        pickle.dump(parameters, _file)

    Ds = df["A"].unique()
    Ws = df["W"].unique()
    association_dw = numpy.empty((len(participants), len(Ds), len(Ws), 3))

    for np, participant in enumerate(participants):
        df_part = df.filter(polars.col("participant") == participant)
        for nd, _d in enumerate(Ds):
            df_d = df_part.filter(polars.col("A") == _d)
            for nw, _w in enumerate(Ws):
                df_dw = df_d.filter(polars.col("W") == _w)

                x = numpy.array(df_dw["IDe(2d)"])
                y = numpy.array(df_dw["Duration"])

                vec = [
                    pearsonr(x, y).statistic,
                    spearmanr(x, y).statistic,
                    kendalltau(x, y).statistic,
                ]
            association_dw[np, nd, nw, :] = vec
    with open("jgp_association_dw.pkl", "wb") as _file:
        pickle.dump(association_dw, _file)


with open("jgp_association_dw.pkl", "rb") as _file:
    association_dw = pickle.load(_file)


Darray = numpy.empty(shape=(association_dw.shape))
Warray = numpy.empty(shape=(association_dw.shape))
assarray = numpy.empty(shape=(association_dw.shape))
Parray = numpy.empty(shape=(association_dw.shape))
for nd, _d in enumerate([256, 512, 1024, 1408]):
    Darray[:, nd, :, :] = _d
for nw, _w in enumerate([64, 96, 128]):
    Warray[:, :, nw, :] = _w
ass_map = {0: "PearsonR", 1: "SpearmanR", 2: "KendallT"}
for nass, ass in enumerate(ass_map.keys()):
    assarray[:, :, :, nass] = ass
for p in range(14):
    Parray[p, :, :, :] = p

_array = numpy.concatenate(
    [
        association_dw.reshape(-1, 1),
        Parray.reshape(-1, 1),
        Darray.reshape(-1, 1),
        Warray.reshape(-1, 1),
        assarray.reshape(-1, 1),
    ],
    axis=1,
)
df = pandas.DataFrame(_array, columns=["value", "P", "D", "W", "association"])
df["association"] = df["association"].map(ass_map)

import statsmodels.api as sm
import statsmodels.formula.api as smf

fig, axs = plt.subplots(1, 3)
for nax, ax in enumerate(axs):
    df_red = df[df["association"] == list(ass_map.values())[nax]]
    seaborn.scatterplot(data=df_red, x="W", y="value", hue="D", ax=ax)
    print("==============================")
    print(df_red["association"].unique())
    md = smf.mixedlm(formula="value ~ D*W", data=df_red, groups=df_red["P"])
    mdf = md.fit()
    print(mdf.summary().as_latex())


plt.ion()
plt.show()
exit()
with open("jgp_parameters.pkl", "rb") as _file:
    parameters = pickle.load(_file)

fig, ax = plt.subplots(1, 1)
ax.hist(parameters[..., 3].ravel(), density=True, alpha=0.5, label=r"$\lambda_0$")
ax.hist(parameters[..., 4].ravel(), density=True, alpha=0.5, label=r"$\lambda_1$")
ax.legend()
plt.show()


with open("jgp_association.pkl", "rb") as _file:
    association = pickle.load(_file)

rows = []
for np, p in enumerate(association):
    for ni, i in enumerate(p):
        rows.append([np, ni, *i])

df = pandas.DataFrame(rows, columns=["participant", "iteration", "r", "rho", "tau"])

exit()


r_icc = pingouin.intraclass_corr(
    data=df, raters="iteration", targets="participant", ratings="r"
)
rho_icc = pingouin.intraclass_corr(
    data=df, raters="iteration", targets="participant", ratings="rho"
)
tau_icc = pingouin.intraclass_corr(
    data=df, raters="iteration", targets="participant", ratings="tau"
)

with open(".tmp.txt", "w") as _file:
    _file.write(r_icc.to_latex())
    _file.write(rho_icc.to_latex())
    _file.write(tau_icc.to_latex())

fig, ax = plt.subplots(1, 1)
seaborn.scatterplot(df, y="rho", x="participant", ax=ax)
plt.ion()
plt.show()


fig, ax = plt.subplots(1, ncols=1)
seaborn.histplot(df, x="r", ax=ax, label="Pearson r")
seaborn.histplot(df, x="rho", ax=ax, label=r"Spearman $\rho$")
seaborn.histplot(df, x="tau", ax=ax, label=r"Kendall $\tau$")
ax.legend()
ax.set_xlabel("Association measure")
plt.tight_layout()
# fig.savefig("img/association.pdf")


### visually nice fits, AIC confirms widely better than OLS
# lets model the parameters a bit
fig, axs = plt.subplots(1, 4, figsize=(20, 5))

exit()
beta = numpy.array([params["beta0"], params["beta1"]]).T
axs[0].scatter(*beta.T)
mean, cov = multivariate_normal.fit(beta)
eigenvalues, eigenvectors = numpy.linalg.eigh(cov)
angle = numpy.degrees(numpy.arctan2(*eigenvectors[:, 0][::-1]))
width, height = 2 * numpy.sqrt(5.991 * eigenvalues)
ellipse = Ellipse(
    xy=mean, width=width, height=height, angle=angle, edgecolor="r", fc="None", lw=2
)
axs[0].add_patch(ellipse)
axs[0].axis("equal")
axs[0].set_xlabel(r"$\beta_0 \text{ (s)}$")
axs[0].set_ylabel(r"$\beta_1 \text{ (s/bit)}$")

axs[1].hist(params["sigma"])
axs[1].set_xlabel(r"$\sigma$")
axs[1].set_ylabel("counts")


expo = numpy.array([params["expo0"], params["expo1"]]).T
axs[2].scatter(*expo.T)
expomean, expocov = multivariate_normal.fit(expo)
eigenvalues, eigenvectors = numpy.linalg.eigh(expocov)
angle = numpy.degrees(numpy.arctan2(*eigenvectors[:, 0][::-1]))
width, height = 2 * numpy.sqrt(5.991 * eigenvalues)
ellipse = Ellipse(
    xy=expomean, width=width, height=height, angle=angle, edgecolor="r", fc="None", lw=2
)
axs[2].add_patch(ellipse)
axs[2].axis("equal")
axs[2].set_xlabel(r"$\lambda_0$")
axs[2].set_ylabel(r"$\lambda_1$")

axs[3].hist(params["expo1"], density=True, alpha=0.6)
axs[3].set_xlabel(r"$\lambda_1$")
axs[3].set_ylabel("Density")
seaborn.kdeplot(params["expo1"], color="k", linewidth=2, ax=axs[3], label="KDE fit")
from scipy.stats import norm as _norm

mu, std = _norm.fit(params["expo1"])  # mu = 0.098, std = 0.017
xmin, xmax = axs[3].get_xlim()
x = numpy.linspace(xmin, xmax, 100)
p = _norm.pdf(x, mu, std)
axs[3].plot(x, p, "r", linewidth=2, label="Gaussian fit")
axs[3].legend(loc=1)

fig.tight_layout()
fig.savefig("img/emg_fits_pop.pdf")
plt.ion()
plt.show()
exit()

#### Consider following model
_dict = {
    "beta": {"type": "gaussian", "params": {"mean": mean, "cov": cov}},
    "sigma": {"type": "uniform", "params": {"min": 0.04, "max": 0.09}},
    "expo": {"type": "uniform", "params": {"min": [0, 0.06], "max": [0.12, 0.15]}},
}
import pickle

with open("JGP_params.pkl", "wb") as _file:
    pickle.dump(_dict, _file)
