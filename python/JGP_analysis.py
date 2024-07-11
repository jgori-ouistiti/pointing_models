from emgregs import emg_reg_heterosked
import polars
import numpy
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import statsmodels.api as sm
from matplotlib.patches import Ellipse
import seaborn

plt.style.use("fivethirtyeight")


class UnitData:
    def __init__(self, df):
        self.df = df

    def __str__(self):
        return self.df.__str__()

    def __repr__(self):
        return self.df.__repr__()

    def __len__(self):
        return self.df.__len__()

    def fit_emgreg(self, id="nominal"):
        if id == "nominal":
            x = self.id_long()
        y = self.mt()
        return emg_reg_heterosked(numpy.array(x), numpy.array(y), maxit=1000)


class GJP_dataset_handler(UnitData):
    def __init__(self, df):
        super().__init__(df)

    def id(self):
        if len(self.df["A"].unique()) != 1:
            raise ValueError("Multiple D conditions in this set")
        if len(self.df["W"].unique()) != 1:
            raise ValueError("Multiple W conditions in this set")
        self._id = numpy.log2(1 + self.df["A"][0] / self.df["W"][0])
        return self._id

    def mean_mt(self):
        self._mean_mt = self.df["MT"].mean()
        return self._mean_mt

    def id_long(self):
        _id = self.id()
        return [_id for i in range(len(self.df))]

    def mt(self):
        return self.df["MT"]


df = polars.read_csv("fitts_GJP.csv", has_header=True, ignore_errors=True)
participants = df["Participant"].unique().to_list()
D_factor = df["A"].unique().to_list()
W_factor = df["W"].unique().to_list()

fig, axs = plt.subplots(4, 4, figsize=(15, 9), sharex=True, sharey=True)


experiment_data = {}
for participant in participants:
    if participant <= 2:
        continue
    participant_data = {}
    p = df.filter(polars.col("Participant") == participant)
    for D in D_factor:
        D_df = p.filter(polars.col("A") == D)
        D_data = {}
        for W in W_factor:
            W_data = D_df.filter(polars.col("W") == W)
            D_data[W] = GJP_dataset_handler(W_data)
            axs[participant // 4, participant % 4].scatter(
                D_data[W].id_long(), D_data[W].mt(), s=2
            )
            axs[participant // 4, participant % 4].scatter(
                D_data[W].id(), D_data[W].mean_mt(), s=10
            )

        participant_data[D] = D_data

    experiment_data[participant] = participant_data


## plots make sense, now fit with EMG, and try to have variance evolve with strategy

params = {"beta0": [], "beta1": [], "sigma": [], "expo0": [], "expo1": []}
fontdict = {"fontsize": 14}

for participant, participant_data in experiment_data.items():
    vec_id = []
    vec_mt = []
    for D, D_data in participant_data.items():
        for W, W_data in D_data.items():
            vec_id.extend(W_data.id_long())
            vec_mt.extend(W_data.mt())
    x = [numpy.min(vec_id), numpy.max(vec_id)]
    fit = emg_reg_heterosked(numpy.array(vec_id), numpy.array(vec_mt), maxit=1000)
    params["beta0"].append(fit["beta"][0])
    params["beta1"].append(fit["beta"][1])
    params["sigma"].append(fit["sigma"])
    params["expo0"].append(fit["ex"][0])
    params["expo1"].append(fit["ex"][1])
    X = sm.add_constant(numpy.array(vec_id))
    ols_fit = sm.OLS(numpy.array(vec_mt), X).fit()
    emg_ll = fit["loglik"]
    ols_ll = ols_fit.llf
    # 2k - 2log(ll)
    ols_aic = 2 * 2 - 2 * ols_ll
    emg_aic = 2 * 5 - 2 * emg_ll
    axs[participant // 4, participant % 4].plot(
        x, [fit["beta"][0] + fit["beta"][1] * _x for _x in x], "r-"
    )
    axs[participant // 4, participant % 4].set_title(
        f"AIC difference (OLS-EMG): {ols_aic- emg_aic:.1f}", fontdict=fontdict
    )

axs[3, 0].set_xlabel("ID (bit)")
axs[3, 0].set_ylabel("MT (s)")
fig.tight_layout()
fig.savefig("img/emg_jgp.pdf")
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
