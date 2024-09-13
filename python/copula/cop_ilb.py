from emgregs import emg_reg_heterosked
import polars
import numpy
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, gaussian_kde
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


# participant = 3
import pandas
from copulas.multivariate import GaussianMultivariate
from copulas.univariate.gaussian import GaussianUnivariate
from copulas.univariate.gamma import GammaUnivariate

num_samples = 100
correlation_copula = []
correlation_an_copula = []
fig, axs = plt.subplots(nrows=4, ncols=13)
fig_an, axs_an = plt.subplots(nrows=4, ncols=13)

for npart, participant in enumerate(experiment_data.keys()):
    rows = []

    participant_data = experiment_data[participant]
    for key, value in participant_data.items():
        for kkey, vvalue in value.items():
            _id = numpy.log2(1 + key / kkey)
            mt = list(vvalue.df["MT"])
            for _mt in mt:
                rows.append([_id, _mt])

    df = pandas.DataFrame(rows, columns=["id", "mt"])
    axs[0, npart].plot(df["id"], df["mt"], "*")
    axs_an[0, npart].plot(df["id"], df["mt"], "*")
    # from copulas.datasets import sample_trivariate_xyz
    # data = sample_trivariate_xyz()
    # copula = GaussianMultivariate({'id': "Gaussian"})
    copula = GaussianMultivariate()

    copula_an = GaussianMultivariate({"id": GaussianUnivariate, "mt": GammaUnivariate})

    print(" ################## start fitting copula ################")
    copula.fit(df)
    print(" ################## start fitting copula ANALYTIC ################")
    copula_an.fit(df)

    if not copula.univariates[0].fitted or not copula.univariates[1].fitted:
        raise RuntimeError("Model not fitted")

    id_min, id_max = df["id"].min(), df["id"].max()
    x_abs = numpy.linspace(id_min, id_max, 1000)
    y = [copula.univariates[0].pdf(x) for x in x_abs]
    y_an = [copula_an.univariates[0].pdf(x) for x in x_abs]

    cop_name = copula.univariates[0].to_dict()["type"]

    axs[1, npart].plot(x_abs, y, "-", label=cop_name)
    axs_an[1, npart].plot(x_abs, y_an, "-", label="GaussianUnivariate")

    mt_min, mt_max = df["mt"].min(), df["mt"].max()
    x_abs = numpy.linspace(mt_min, mt_max, 1000)
    y = [copula.univariates[1].pdf(x) for x in x_abs]
    y_an = [copula_an.univariates[1].pdf(x) for x in x_abs]
    cop_name = copula.univariates[1].to_dict()["type"]
    axs[2, npart].plot(x_abs, y, "-", label=cop_name)
    axs_an[2, npart].plot(x_abs, y_an, "-", label="GammaUnivariate")

    conditions = {
        "id": pandas.Series(
            [id_min for i in range(15)]
            + [(id_min + id_max) / 2 for i in range(15)]
            + [id_max for i in range(15)]
        )
    }
    synthetic_data = copula.sample(num_samples)
    axs[3, npart].plot(synthetic_data["id"], synthetic_data["mt"], "*", label="sampled")
    synthetic_data_an = copula_an.sample(num_samples)
    axs_an[3, npart].plot(
        synthetic_data_an["id"], synthetic_data_an["mt"], "*", label="sampled"
    )

    correlation_copula.append(numpy.array(copula.correlation)[0][1])
    correlation_an_copula.append(numpy.array(object=copula_an.correlation)[0][1])


figkde, axkde = plt.subplots(1, 1)
data = correlation_copula
kde = gaussian_kde(data)
x_values = numpy.linspace(min(data), max(data), 1000)
kde_values = kde(x_values)
axkde.plot(x_values, kde_values, "-", label="empirical")
data = correlation_an_copula
kde = gaussian_kde(data)
x_values = numpy.linspace(min(data), max(data), 1000)
kde_values = kde(x_values)
axkde.plot(x_values, kde_values, "-", label="analytical")

plt.ion()
plt.show()
