from emgregs import emg_reg_heterosked
import polars
import numpy
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import statsmodels.api as sm
from matplotlib.patches import Ellipse
import seaborn
import pingouin

import warnings

warnings.filterwarnings("ignore")


plt.style.use("fivethirtyeight")


parameters = numpy.empty((15, 6, 5))

for np, participant in enumerate(range(1, 16)):
    for ni, iteration in enumerate(range(6)[2:]):

        df_filt = polars.read_csv(f"dump/{participant}_{iteration}.csv")
        # try:
        fit = emg_reg_heterosked(
            numpy.array(df_filt["IDe(2d)"]),
            numpy.array(df_filt["Duration"]),
            beta=[0, 0.5],
            maxit=1000,
            basinhopping=True,
        )

        fig, ax = plt.subplots(1, 1)
        X = sm.add_constant(numpy.array(df_filt["IDe(2d)"]))
        ols_fit = sm.OLS(numpy.array(df_filt["Duration"]), X).fit()
        emg_ll = fit["loglik"]
        ols_ll = ols_fit.llf
        # 2k - 2log(ll)
        ols_aic = 2 * 2 - 2 * ols_ll
        emg_aic = 2 * 5 - 2 * emg_ll
        x = [
            numpy.min(numpy.asarray(df_filt["IDe(2d)"])),
            numpy.max(numpy.asarray(df_filt["IDe(2d)"])),
        ]
        ax.plot(numpy.array(df_filt["IDe(2d)"]), numpy.array(df_filt["Duration"]), "o")
        ax.plot(x, [fit["beta"][0] + fit["beta"][1] * _x for _x in x], "r-")
        ax.set_title(
            f"{participant},{iteration}: AIC difference (OLS-EMG): {ols_aic- emg_aic:.1f}",
        )
        if participant == 1 and iteration == 3:
            print(fit["res"])
            plt.show()
            exit()
        fig.savefig(f"tmp/emg_{participant}_{iteration}.pdf")
        # except ValueError:
        #     fit = dict(
        #         beta=[numpy.nan, numpy.nan], sigma=numpy.nan, ex=[numpy.nan, numpy.nan]
        #     )
        plt.close(fig)
        params = [
            fit["beta"][0],
            fit["beta"][1],
            fit["sigma"],
            fit["ex"][0],
            fit["ex"][1],
        ]
        parameters[np, ni, :] = params

    continue

### visually nice fits, AIC confirms widely better than OLS
# lets model the parameters a bit
fig, axs = plt.subplots(1, 4, figsize=(20, 5))


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
