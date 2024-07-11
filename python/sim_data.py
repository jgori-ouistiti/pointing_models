import emgregs
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
import numpy
from matplotlib.patches import Ellipse
import polars

plt.style.use("fivethirtyeight")


def gen_emg_data_strategy_no_correlation_control(
    strategies=[-1, -0.5, 0, 0.5, 1], nparticipants=12, nblocks=5, ntrials=15
):

    rows = []
    # p x strat x block x trials x ide x mt

    # participant distributions:
    beta_rv = stats.multivariate_normal(
        mean=numpy.array([0.07823552, 0.20838737]),
        cov=numpy.array([[0.00323031, -0.00136086], [-0.00136086, 0.0010003]]),
    )
    sigma_rv = stats.uniform(loc=0.04, scale=0.05)
    lambda_emg_rv = stats.norm(loc=0.1, scale=0.02)

    xp_data = {}
    for participant in range(nparticipants):
        # sample parameters
        beta = beta_rv.rvs(1)
        sigma = sigma_rv.rvs(1)
        lambda_emg = (0.05, lambda_emg_rv.rvs(1)[0])
        p_data = {}
        for ns, strat in enumerate(strategies):
            mu = 4.73 + strat * 2.18
            std = 1.03 + 0.28 * strat

            ide_distrib = stats.norm(loc=mu, scale=std)
            ide = ide_distrib.rvs(nblocks)

            X, Y = [], []

            for nb, block in enumerate(ide):
                _input = numpy.ones((ntrials, 2))
                _input[:, 1] = [block for b in range(ntrials)]
                x, y = emgregs.sim_emg_reg_heterosked(
                    X=_input,
                    n=ntrials,
                    beta=beta,
                    sigma=sigma,
                    expo_scale=lambda_emg,
                ).values()
                x, y = x.squeeze(), y.squeeze()

                X.append(x)
                Y.append(y)
                for n, (_x, _y) in enumerate(zip(x, y)):
                    rows.append([participant, ns, nb, n, _x, _y])

            p_data[strat] = (X, Y)
        xp_data[participant] = p_data

    df = polars.DataFrame(
        rows, schema=["participants", "strategy", "block", "trial", "id", "mt"]
    )
    return xp_data, df


def gen_emg_data_strategy_correlation_control(
    strategies=[-1, -0.5, 0, 0.5, 1], nparticipants=15, nblocks=10, ntrials=20
):

    rows = []
    # p x strat x block x trials x ide x mt

    # participant distributions:
    beta_rv = stats.multivariate_normal(
        mean=numpy.array([0.07823552, 0.20838737]),
        cov=numpy.array([[0.00323031, -0.00136086], [-0.00136086, 0.0010003]]),
    )
    sigma_rv = stats.uniform(loc=0.04, scale=0.05)

    xp_data = {}
    for participant in range(nparticipants):
        # sample parameters
        beta = beta_rv.rvs(1)
        sigma = sigma_rv.rvs(1)
        p_data = {}
        for ns, strat in enumerate(strategies):
            mu = numpy.array([4.73 + strat * 2.18, 1.28 + 0.65 * strat])
            crosscovariance = (1.03 + 0.28 * strat) * (0.37 + 0.1 * strat) * 0.36
            covariance = numpy.array(
                [
                    [(1.03 + 0.28 * strat) ** 2, crosscovariance],
                    [crosscovariance, (0.37 + 0.1 * strat) ** 2],
                ]
            )
            bivariate_gaussian = stats.multivariate_normal(mean=mu, cov=covariance)

            ide, mt = bivariate_gaussian.rvs(nblocks).T

            X, Y = [], []

            for nb, (_id, _mt) in enumerate(zip(ide, mt)):
                _input = numpy.ones((ntrials, 2))
                _input[:, 1] = [_id for b in range(ntrials)]
                lambda_emg = (0.1, max(0.01, (_mt - beta[0] - beta[1] * _id) / _id))
                x, y = emgregs.sim_emg_reg_heterosked(
                    X=_input,
                    n=ntrials,
                    beta=beta,
                    sigma=sigma,
                    expo_scale=lambda_emg,
                ).values()
                x, y = x.squeeze(), y.squeeze()

                X.append(x)
                Y.append(y)
                for n, (_x, _y) in enumerate(zip(x, y)):
                    rows.append([participant, ns, nb, n, _x, _y])

            p_data[strat] = (X, Y)
        xp_data[participant] = p_data

    df = polars.DataFrame(
        rows, schema=["participants", "strategy", "block", "trial", "id", "mt"]
    )
    return xp_data, df


xp_data, df = gen_emg_data_strategy_correlation_control(
    strategies=[-1, -0.5, 0, 0.5, 1], nparticipants=15, nblocks=10, ntrials=20
)

df_dict = {group: data for group, data in df.group_by("strategy")}
ev_dict = {}
for key, value in df_dict.items():
    x, y = value.group_by(["block", "participants"]).mean().select(["id", "mt"])
    ev_dict[key] = (x, y)


from eval_data import *
import seaborn

g1, g2 = evaluate_data_mean_mvgauss(ev_dict, strategies=[-1, -0.5, 0, 0.5, 1])
plt.tight_layout()

fig, ax = plt.subplots(1, 1)
seaborn.scatterplot(df, x="id", y="mt", hue="strategy")


xp_data, df = gen_emg_data_strategy_no_correlation_control(
    strategies=[-1, -0.5, 0, 0.5, 1], nparticipants=15, nblocks=10, ntrials=20
)

df_dict = {group: data for group, data in df.group_by("strategy")}
ev_dict = {}
for key, value in df_dict.items():
    x, y = value.group_by(["block", "participants"]).mean().select(["id", "mt"])
    ev_dict[key] = (x, y)


from eval_data import *
import seaborn

g1, g2 = evaluate_data_mean_mvgauss(ev_dict, strategies=[-1, -0.5, 0, 0.5, 1])
plt.tight_layout()

fig, ax = plt.subplots(1, 1)
seaborn.scatterplot(df, x="id", y="mt", hue="strategy")

plt.ion()
plt.show()
