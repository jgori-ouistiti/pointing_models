import numpy
import scipy.stats as stats
import seaborn
import matplotlib.pyplot as plt


from scipy.stats._continuous_distns import exponnorm_gen


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


x = gen_gaussian_copula(1000, 0.36, viz=False)
norm = stats.norm()
x_unif = norm.cdf(x)


mu_emg = 1
sigma_emg = 0.1
lambda_emg = 1

fig, axs = plt.subplots(1, 5)
for ns, strat in enumerate(strategies):
    print(f"================={ns}")
    loc_m1 = 4.73 + strat * 2.18
    scale_m1 = 1.03 + 0.28 * strat

    # m1 = stats.norm(loc=loc_m1, scale=scale_m1)
    # m2 = stats.exponnorm(1 / (sigma_emg * lambda_emg), loc=mu_emg, scale=sigma_emg)

    # N = 1000

    # x1_trans = m1.ppf(x_unif[:N, 0])
    # x2_trans = m2.ppf(x_unif[:N, 1])

    # h = seaborn.jointplot(x=x1_trans, y=x2_trans, kind="kde")
    # h = seaborn.jointplot(x=x1_trans, y=x2_trans, kind="scatter", joint_kws={"s": 4})

    # ==================  conditional marginal version

    mu_emg = numpy.array([0.1, 0.2])
    lambda_emg = numpy.array([0.05, 0.1])
    m1 = stats.norm(loc=loc_m1, scale=scale_m1)
    x1_trans = m1.ppf(x_unif[:, 0])
    x2_trans = []
    for x, y in zip(x1_trans, x_unif[:, 1]):
        X = numpy.array([1, x])
        _lambda = numpy.dot(lambda_emg, X)
        _mu = numpy.dot(mu_emg, X)
        m2 = stats.exponnorm(1 / (sigma_emg * _lambda), loc=_mu, scale=sigma_emg)
        x2_trans.append(m2.ppf(y))

    # h = seaborn.jointplot(x=x1_trans, y=x2_trans, kind="kde")
    h = seaborn.jointplot(
        x=x1_trans, y=x2_trans, kind="scatter", joint_kws={"s": 4}, ax=axs[ns]
    )
plt.show()
