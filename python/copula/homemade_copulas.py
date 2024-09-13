import numpy
import scipy.stats as stats
import seaborn
import matplotlib.pyplot as plt
import emgregs
from tqdm import tqdm

from scipy.stats._continuous_distns import exponnorm_gen


mvnorm = stats.multivariate_normal(mean=[0, 0], cov=[[1.0, 0.5], [0.5, 1.0]])
x = mvnorm.rvs(1000)

norm = stats.norm()
x_unif = norm.cdf(x)

# m1 = stats.gumbel_l()
# m2 = stats.beta(a=10, b=2)

###  ===========================  simple copula works
# strat = 0
# loc_m1 = 4.73 + strat * 2.18
# scale_m1 = 1.03 + 0.28 * strat
# mu_emg = 1
# sigma_emg = 0.1
# lambda_emg = 1

# m1 = stats.norm(loc=loc_m1, scale=scale_m1)
# m2 = stats.exponnorm(1 / (sigma_emg * lambda_emg), loc=mu_emg, scale=sigma_emg)

# x1_trans = m1.ppf(x_unif[:, 0])
# x2_trans = m2.ppf(x_unif[:, 1])

# mean, cov = stats.norm.fit(x1_trans)
# res = stats.exponnorm.fit(x2_trans)

# h = seaborn.jointplot(x=x1_trans, y=x2_trans, kind="kde", stat_func=None)

# x1 = m1.rvs(1000)
# x2 = m2.rvs(1000)
# h = seaborn.jointplot(x=x1, y=x2, kind="kde", stat_func=None)
# plt.show()

### ============================ block-based copula naive approach
# rho = 0.36
# ntrials = 5

# mean = [0] + [0 for i in range(ntrials)]
# cov = numpy.diag([1.0 for i in range(ntrials + 1)])
# cov[:, 0] = rho
# cov[0, :] = rho
# cov[0, 0] = 1

# mvnorm = stats.multivariate_normal(mean=mean, cov=cov)

# x = mvnorm.rvs(1000)
# norm = stats.norm()
# x_unif = norm.cdf(x)

# strat = 0
# loc_m1 = 4.73 + strat * 2.18
# scale_m1 = 1.03 + 0.28 * strat
# mu_emg = 1
# sigma_emg = 0.1
# lambda_emg = 1

# m1 = stats.norm(loc=loc_m1, scale=scale_m1)
# m2 = stats.exponnorm(1 / (sigma_emg * lambda_emg), loc=mu_emg, scale=sigma_emg)

# x1_trans = m1.ppf(x_unif[:, 0])
# ytrans = []
# for i in range(ntrials):
#     ytrans.append(m2.ppf(x_unif[:, i + 1]))

# X = []
# Y = []
# for i in range(ntrials):
#     X += list(x1_trans)
#     Y += list(ytrans[i])

# mean, cov = stats.norm.fit(x1_trans)
# for i in range(ntrials):
#     res = stats.exponnorm.fit(ytrans[i])
#     print(res)


# h = seaborn.jointplot(x=X, y=Y, kind="kde", stat_func=None)

# seaborn.scatterplot(x=X, y=Y)

# plt.show()

####### ======================= conditional mean version

rho = 0.8
ntrials = 5
copula_size = 1000

mean = 0
cov = 1

norm_rv = stats.norm(loc=0, scale=1)

x = numpy.zeros((ntrials + 1, copula_size))
x[0, :] = norm_rv.rvs(copula_size)
for nx, _x in enumerate(x[0, :]):
    loc = 0 + rho / 1 * (_x - 0)
    scale = numpy.sqrt(1 - rho**2 / 1)
    _norm_rv = stats.norm(loc=loc, scale=scale)
    x[1:, nx] = _norm_rv.rvs(ntrials)

x = x.T

X = []
Y = []
for nx in range(ntrials):
    X += list(x[0, :])
    Y += list(x[nx + 1, :])


norm = stats.norm()
x_unif = norm.cdf(x)

strat = 0
loc_m1 = 4.73 + strat * 2.18
scale_m1 = 1.03 + 0.28 * strat
mu_emg = 1
sigma_emg = 0.1
lambda_emg = 1

m1 = stats.norm(loc=loc_m1, scale=scale_m1)
m2 = stats.exponnorm(1 / (sigma_emg * lambda_emg), loc=mu_emg, scale=sigma_emg)

x1_trans = m1.ppf(x_unif[:, 0])
ytrans = []
for i in range(ntrials):
    ytrans.append(m2.ppf(x_unif[:, i + 1]))

X = []
Y = []
for i in range(ntrials):
    X += list(x1_trans)
    Y += list(ytrans[i])

mean, cov = stats.norm.fit(x1_trans)
for i in range(ntrials):
    res = stats.exponnorm.fit(ytrans[i])
    print(res)


h = seaborn.jointplot(x=X, y=Y, kind="kde", stat_func=None)

seaborn.scatterplot(x=X, y=Y)

plt.show()

####### ======================= conditional mean version with conditional marginal and strategies
# strategies = [-1, -0.5, 0, 0.5, 1]

# rho = 0.76
# ntrials = 25
# copula_size = 1000

# mean = 0
# cov = 1

# norm_rv = stats.norm(loc=0, scale=1)

# x = numpy.zeros((ntrials + 1, copula_size))
# x[0, :] = norm_rv.rvs(copula_size)
# for nx, _x in enumerate(x[0, :]):
#     loc = 0 + rho / 1 * (_x - 0)
#     scale = numpy.sqrt(1 - rho**2 / 1)
#     _norm_rv = stats.norm(loc=loc, scale=scale)
#     x[1:, nx] = _norm_rv.rvs(ntrials)

# x = x.T

# X = []
# Y = []
# for nx in range(ntrials):
#     X += list(x[0, :])
#     Y += list(x[nx + 1, :])


# norm = stats.norm()
# x_unif = norm.cdf(x)


# _dict = {}
# _dict_array = {}
# fig, axs = plt.subplots(1, 5)
# for ns, strat in tqdm(enumerate(strategies)):
#     Xout = []
#     Yout = []
#     loc_m1 = 4.73 + strat * 2.18
#     scale_m1 = 1.03 + 0.28 * strat
#     # mu_emg = numpy.array([0.1, 0.2])
#     # lambda_emg = numpy.array([0.05, 0.1])
#     # lambda_emg = numpy.array([.7, 0])
#     lambda_emg = numpy.array([0.37 + 0.1 * strat, 0.1])
#     mu_emg = numpy.array([1.08 + 0.68 * strat - (0.37 + 0.1 * strat), 0])
#     sigma_emg = 0.05
#     m1 = stats.norm(loc=loc_m1, scale=scale_m1)
#     x1_trans = m1.ppf(x_unif[:, 0])
#     x2_trans = []
#     N_copula = len(x1_trans)
#     _array = numpy.zeros((len(x1_trans), ntrials + 1))
#     # indexes = numpy.random.choice(numpy.arange(N_copula), N, replace=False)
#     for nt in tqdm(range(ntrials)):
#         for niter, (x, y) in enumerate(zip(x1_trans, x_unif[:, 1 + nt])):
#             X = numpy.array([1, x])
#             _lambda = numpy.dot(lambda_emg, X)
#             _mu = numpy.dot(mu_emg, X)
#             m2 = stats.exponnorm(1 / (sigma_emg * _lambda), loc=_mu, scale=sigma_emg)
#             Xout.append(x)
#             Yout.append(m2.ppf(y))
#             _array[niter, nt + 1] = m2.ppf(y)
#             _array[niter, 0] = x
#             # x2_trans.append(m2.ppf(y))
#     _dict[ns] = (Xout, Yout)
#     _dict_array[ns] = _array

# fig, axs = plt.subplots(1, 5)
# for ni, i in enumerate(strategies):
#     h = seaborn.jointplot(x=_dict[ni][0], y=_dict[ni][1], kind="kde", stat_func=None)
#     seaborn.scatterplot(x=_dict[ni][0], y=_dict[ni][1], ax=axs[ni])


# rng = numpy.random.default_rng(seed=1234)
# import polars

# container = numpy.zeros((5, 10, 5))
# rows = []
# for key, value in _dict_array.items():
#     for np, p in enumerate(range(10)):
#         for nb, b in enumerate(range(5)):
#             values = rng.choice(value, size=1, axis=0).squeeze()
#             for v in values[1:]:
#                 rows.append([key, np, nb, values[0], v])

# df = polars.DataFrame(rows, schema=["strategy", "participants", "block", "id", "mt"])
# from eval_data import *

# handles = evaluate_data_mean_mvgauss_df(df=df)
# plt.ion()
# plt.show()

# plt.show()
# exit()

strategies = [-1, -0.5, 0, 0.5, 1]
strat = strategies[0]


def gen_gaussian_copula(N, correlation, viz=False):
    mu = numpy.array([0, 0])
    covariance = numpy.array([[1, correlation], [correlation, 1]])

    bivariate_gaussian = stats.multivariate_normal(mean=mu, cov=covariance)
    x = bivariate_gaussian.rvs(N)
    if not viz:
        return x

    h = seaborn.jointplot(x=x[:, 0], y=x[:, 1], kind="kde", stat_func=None)
    h.set_axis_labels("X1", "X2", fontsize=16)
    return x, h


N_copula = 1000
N = 1000
x_copula = gen_gaussian_copula(N_copula, 0.36, viz=False)
# x_copula = gen_gaussian_copula(N_copula, 0.95, viz=False)
# x_copula = gen_gaussian_copula(N_copula, 0, viz=False)


mu_emg = 1
sigma_emg = 0.1
lambda_emg = 1

N = 1000

_dict = {}
fig, axs = plt.subplots(1, 5)
for ns, strat in enumerate(strategies):
    indx = numpy.random.choice(numpy.arange(N_copula), N)
    x = x_copula[indx, :]
    norm = stats.norm()
    x_unif = norm.cdf(x)
    print(f"================={ns}")
    loc_m1 = 4.73 + strat * 2.18
    scale_m1 = 1.03 + 0.28 * strat

    m1 = stats.norm(loc=loc_m1, scale=scale_m1)
    m2 = stats.exponnorm(1 / (sigma_emg * lambda_emg), loc=mu_emg, scale=sigma_emg)

    x1_trans = m1.ppf(x_unif[:, 0])
    x2_trans = m2.ppf(x_unif[:, 1])

    h = seaborn.jointplot(x=x1_trans, y=x2_trans, kind="kde")
    h = seaborn.jointplot(x=x1_trans, y=x2_trans, kind="scatter", joint_kws={"s": 4})

    # ==================  conditional marginal version

    # mu_emg = numpy.array([0.1, 0.2])
    # lambda_emg = numpy.array([0.05, 0.1])
    # m1 = stats.norm(loc=loc_m1, scale=scale_m1)
    # x1_trans = m1.ppf(x_unif[:, 0])
    # x2_trans = []
    # N_copula = len(x1_trans)
    # indexes = numpy.random.choice(numpy.arange(N_copula), N, replace=False)
    # for x, y in zip(x1_trans, x_unif[:, 1]):
    #     X = numpy.array([1, x])
    #     _lambda = numpy.dot(lambda_emg, X)
    #     _mu = numpy.dot(mu_emg, X)
    #     m2 = stats.exponnorm(1 / (sigma_emg * _lambda), loc=_mu, scale=sigma_emg)
    #     x2_trans.append(m2.ppf(y))

    # # h = seaborn.jointplot(x=x1_trans, y=x2_trans, kind="kde")
    # h = seaborn.jointplot(
    #     x=x1_trans, y=x2_trans, kind="scatter", joint_kws={"s": 4}, ax=axs[ns]
    # )
    # x2_trans = numpy.array(x2_trans)

    _dict[ns] = (x1_trans, x2_trans)

rng = numpy.random.default_rng(seed=1234)
import polars

container = numpy.zeros((5, 10, 5))
rows = []
for key, value in _dict.items():
    for np, p in enumerate(range(10)):
        for nb, b in enumerate(range(5)):
            x, y = value
            index = rng.choice(len(x), 100)
            # container[key, np, nb] = x[index].mean(), y[index].mean()
            for idx in index:
                rows.append([key, np, nb, x[idx], y[idx]])

df = polars.DataFrame(rows, schema=["strategy", "participants", "block", "id", "mt"])
from eval_data import *

handles = evaluate_data_mean_mvgauss_df(df=df)
plt.ion()
plt.show()
