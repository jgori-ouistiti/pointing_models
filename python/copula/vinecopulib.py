import pyvinecopulib
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
