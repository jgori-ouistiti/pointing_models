from emgregs import emg_reg_heterosked
import polars
import numpy
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, gaussian_kde
import statsmodels.api as sm
from matplotlib.patches import Ellipse
import seaborn
import pandas


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

        participant_data[D] = D_data

    experiment_data[participant] = participant_data


fig, axs = plt.subplots(2, 1)

participant = 3
rows = []

participant_data = experiment_data[participant]
for key, value in participant_data.items():
    for kkey, vvalue in value.items():
        _id = numpy.log2(1 + key / kkey)
        mt = list(vvalue.df["MT"])
        for _mt in mt:
            rows.append([_id, _mt])

df = pandas.DataFrame(rows, columns=["norm", "gamma"])
axs[0].plot(df["norm"], df["gamma"], "*")


df.to_csv("copula_data.csv")

from scipy.stats import norm, multivariate_normal

# Define the Gaussian copula parameters
rho = 0.8  # correlation coefficient
mean = [0, 0]
cov = [[1, rho], [rho, 1]]

# Generate samples from a bivariate normal distribution
# num_samples = 1000
# samples = multivariate_normal.rvs(mean=mean, cov=cov, size=num_samples)


# Define the grid for the contour plot
u = numpy.linspace(0, 1, 100)
v = numpy.linspace(0, 1, 100)
U, V = numpy.meshgrid(u, v)

# Evaluate the Gaussian copula density function on the grid
pos = numpy.dstack((U, V))
rv = multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]])
Z = rv.pdf(norm.ppf(pos))

# Plot the contour plot
fig, axs = plt.subplots(1, 2)
contour = axs[0].contour(U, V, Z, levels=10, cmap="viridis")

# rotated tawn type 1 180 degrees


def rotated_tawn_cdf(u, v, theta, delta):
    # delta in [0,1]
    # theta in R+

    cdf_uv = (
        u
        + v
        - 1
        + (
            (1 - delta)
            + delta * (1 - u) ** theta
            + delta * (1 - v) ** theta
            - delta * (1 - u) ** theta * (1 - v) ** theta
        )
        ** (1 / delta)
    )
    return cdf_uv


def numerical_pdf_rotated_tawn(u, v, theta, delta, epsilon=1e-5):
    cdf_uv = rotated_tawn_cdf(u, v, theta, delta)
    cdf_ueps_v = rotated_tawn_cdf(u + epsilon, v, theta, delta)
    cdf_u_veps = rotated_tawn_cdf(u, v + epsilon, theta, delta)
    cdf_ueps_veps = rotated_tawn_cdf(u + epsilon, v + epsilon, theta, delta)

    pdf_uv = (cdf_ueps_veps - cdf_ueps_v - cdf_u_veps + cdf_uv) / (epsilon**2)
    return pdf_uv


# Z = numpy.zeros(Z.shape)

# for nu, x in enumerate(numpy.linspace(0, 1, 100)):
#     for nv, y in enumerate(numpy.linspace(0, 1, 100)):
#         Z[nv, nu] = numerical_pdf_rotated_tawn(*norm.ppf([x, y]), theta, delta)

theta = 2.81
delta = 0.81
pos = numpy.dstack((U, V))
ppf_pos = norm.ppf(pos)
Z = numerical_pdf_rotated_tawn(ppf_pos[:, :, 0], ppf_pos[:, :, 1], theta, delta)

# Z = rotated_tawn_cdf(U, V, theta, delta)

# Z = numerical_pdf_rotated_tawn(U, V, theta, delta)

contour = axs[1].contour(U, V, Z, levels=10, cmap="viridis")

plt.grid(True)
plt.show()
exit()

from copulae import EmpiricalCopula, pseudo_obs

u = pseudo_obs(df)
emp_cop = EmpiricalCopula(u, smoothing="beta")


# Z = numpy.apply_along_axis(mtcj_wrapper, axis=2, arr=pos)
# contour = axs[1].contour(X, Y, Z, levels=10, cmap="viridis")
