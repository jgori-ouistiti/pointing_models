import matplotlib.pyplot as plt
import seaborn
import numpy
import scipy.stats as stats

plt.style.use("fivethirtyeight")

EMG_params = {"mu": 0.5293452, "sigma": 0.1890695, "lambda": 1.3338371}
loc = EMG_params["mu"]
scale = EMG_params["sigma"]
K = 1 / (EMG_params["sigma"] * EMG_params["lambda"])
emg = stats.distributions.exponnorm(K, loc=loc, scale=scale)
fig, ax = plt.subplots(1, 1, figsize=(3, 3))
seaborn.kdeplot(emg.rvs(10000), ax=ax)
ax.set_xlabel("MT (s)")
ax.set_ylabel("Density")
plt.tight_layout()
plt.ion()
plt.savefig("img/copula_scheme_00.pdf")

ID = stats.distributions.norm(loc=4, scale=0.5)
fig, ax = plt.subplots(1, 1, figsize=(3, 3))
seaborn.kdeplot(ID.rvs(10000), ax=ax)
ax.set_xlabel("ID (bit)")
ax.set_ylabel("Density")
plt.tight_layout()
plt.ion()
plt.savefig("img/copula_scheme_01.pdf")


x = stats.uniform(0, 1).rvs(10000)
mt_icdf = emg.ppf(x)
h = seaborn.jointplot(x=x, y=mt_icdf, ratio=2, space=0)
h.ax_marg_x.set_box_aspect(0.1)
h.ax_marg_x.set_title("Uniform density")
h.ax_marg_y.set_title("EMG density", loc="center", rotation=270, x=1, y=0.5)
h.ax_marg_y.grid(False)
h.ax_marg_x.grid(False)
h.ax_joint.set_xlabel(r"$\mathbb{P}[\text{MT} \leq \text{mt}]$")
h.ax_joint.set_ylabel("mt")
h.figure.set_figheight(4)
h.figure.subplots_adjust(hspace=0, wspace=0)

plt.tight_layout()
plt.ion()
plt.savefig("img/copula_scheme_1.pdf")
plt.show()

x = stats.uniform(0, 1).rvs(10000)
id_icdf = ID.ppf(x)
h = seaborn.jointplot(x=x, y=id_icdf)
h.ax_marg_x.set_title("Uniform density")
h.ax_marg_y.set_title("Normal density", loc="center", rotation=270, x=1, y=0.5)
h.ax_marg_y.grid(False)
h.ax_marg_x.grid(False)
h.ax_joint.set_xlabel(r"$\mathbb{P}[\text{ID} \leq \text{id}]$")
h.ax_joint.set_ylabel("id")
plt.tight_layout()
plt.ion()
plt.savefig("img/copula_scheme_2.pdf")
plt.show()


h = seaborn.jointplot(y=mt_icdf, x=id_icdf)
h.ax_joint.set_xlabel("ID")
h.ax_joint.set_ylabel("MT")
h.ax_marg_y.grid(False)
h.ax_marg_x.grid(False)
plt.tight_layout()
plt.ion()
plt.savefig("img/copula_scheme_3.pdf")
plt.show()


with plt.style.context("./kde.mplstyle"):
    mvnorm = stats.multivariate_normal(mean=[0, 0], cov=[[1.0, 0.44], [0.44, 1.0]])
    x = mvnorm.rvs(size=10000)
    plt.figure()
    h = seaborn.kdeplot(x=x[:, 0], y=x[:, 1], fill=True)
    h.set_xlabel("Variable X")
    h.set_ylabel("Variable Y")
    h.set_title("Gaussian copula (KDE of densities)")
    plt.tight_layout()
    plt.ion()
    plt.savefig("img/copula_scheme_4.pdf")
    plt.show()

norm = stats.norm()
x_unif = norm.cdf(x)
id = ID.ppf(x_unif[:, 0])
mt = emg.ppf(x_unif[:, 1])
h = seaborn.jointplot(x=id, y=mt, kind="scatter")
h.ax_joint.set_xlabel("ID")
h.ax_joint.set_ylabel("MT")
h.ax_marg_y.grid(False)
h.ax_marg_x.grid(False)
plt.tight_layout()
plt.ion()
plt.savefig("img/copula_scheme_5.pdf")
plt.show()
