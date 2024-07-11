import scipy.stats as stats
import numpy
import emgregs
import polars
import seaborn
import matplotlib.pyplot as plt
from tqdm import tqdm


def _gen_emg(beta, sigma, lambda_emg, ide=None, ntrials=50):
    X = numpy.ones((ntrials, 2))
    X[:, 1] = ide
    x, y = emgregs.sim_emg_reg_heterosked(
        X=X,
        n=ntrials,
        beta=beta,
        sigma=sigma,
        expo_scale=lambda_emg,
    ).values()
    return x.squeeze(), y.squeeze()


def gen_emg_data_strategy_no_correlation_control_gaussian_ide(
    beta_per_participant,
    sigma_per_participant,
    lambda_emg_per_participant,
    gaussian_ide_dict,
    nblocks=5,
    ntrials=15,
):

    strategy = gaussian_ide_dict["label"]
    mu_ide = gaussian_ide_dict["mu"]
    std_ide = gaussian_ide_dict["std"]

    rows = []
    # p x strat x block x trials x ide x mt

    # participant distributions:
    xp_data = {}
    for np, (beta, sigma, _lambda) in enumerate(
        zip(beta_per_participant, sigma_per_participant, lambda_emg_per_participant)
    ):
        # sample parameters
        p_data = {}
        for ns, (strat, mu, std) in enumerate(zip(strategies, mu_ide, std_ide)):
            ide_distrib = stats.norm(loc=mu, scale=std)
            ide = ide_distrib.rvs(nblocks)

            X, Y = [], []

            for nb, block in enumerate(ide):
                x, y = _gen_emg(beta, sigma, _lambda, ide=block, ntrials=ntrials)
                X.append(x)
                Y.append(y)
                for nt, (_x, _y) in enumerate(zip(x, y)):
                    rows.append([np, ns, nb, nt, _x, _y])

            p_data[strat] = (X, Y)
        xp_data[np] = p_data

    df = polars.DataFrame(
        rows, schema=["participants", "strategy", "block", "trial", "id", "mt"]
    )
    return xp_data, df


def gen_emg_data_strategy_correlation_control_gaussian_ide(
    beta_per_participant,
    sigma_per_participant,
    lambda_emg_per_participant,
    gaussian_joint_dict,
    nblocks=5,
    ntrials=15,
):

    strategy = gaussian_joint_dict["label"]
    MU = gaussian_joint_dict["mu"]
    covariance = gaussian_joint_dict["covariance"]

    rows = []
    # p x strat x block x trials x ide x mt

    # participant distributions:
    xp_data = {}
    for np, (beta, sigma, _lambda) in enumerate(
        zip(beta_per_participant, sigma_per_participant, lambda_emg_per_participant)
    ):
        # sample parameters
        p_data = {}
        for ns, (strat, mu, cov) in enumerate(zip(strategies, MU, covariance)):
            bivariate_gaussian = stats.multivariate_normal(mean=mu, cov=cov)
            ide, mt = bivariate_gaussian.rvs(nblocks).T

            X, Y = [], []

            for nb, (_id, _mt) in enumerate(zip(ide, mt)):
                lambda_emg = (
                    _lambda[0],
                    max(_lambda[1], (_mt - beta[0] - beta[1] * _id) / _id),
                )  # correct lambda
                x, y = _gen_emg(beta, sigma, lambda_emg, ide=_id, ntrials=ntrials)
                X.append(x)
                Y.append(y)
                for nt, (_x, _y) in enumerate(zip(x, y)):
                    rows.append([np, ns, nb, nt, _x, _y])

            p_data[strat] = (X, Y)
        xp_data[np] = p_data

    df = polars.DataFrame(
        rows, schema=["participants", "strategy", "block", "trial", "id", "mt"]
    )
    return xp_data, df


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


def gen_emg_data_gaussian_copula_strategy(
    gaussian_copula, beta, sigma, lambda_vec, mu, scale, Ntrials
):
    m1 = stats.norm(loc=mu, scale=scale)
    # x1_trans = m1.ppf(gaussian_copula[:N, 0])
    x1_trans = m1.ppf(gaussian_copula[:, 0])
    x2_trans = []
    for x, y in zip(x1_trans, gaussian_copula[:, 1]):
        X = numpy.array([1, x])
        _lambda = numpy.dot(lambda_vec, X)
        _mu = numpy.dot(beta, X)
        m2 = stats.exponnorm(1 / (sigma * _lambda), loc=_mu, scale=sigma)
        x2_trans.append(m2.ppf(y))

    indexes = numpy.random.choice(len(gaussian_copula[:, 1]), Ntrials, replace=False)

    return [x1_trans[index] for index in indexes], [
        x2_trans[index] for index in indexes
    ]  # direct indexing does not work for some reason


def gen_emg_data_gaussian_copula(
    copula_dict, emg_dict, nblocks=5, ntrials=15, viz=False
):
    correlation_m1 = copula_dict["correlation"]
    loc_m1 = copula_dict["mu"]
    scale_m1 = copula_dict["scale"]
    strategies = copula_dict["label"]

    beta_per_participant = emg_dict["beta"]
    sigma_per_participant = emg_dict["sigma"]
    lambda_emg_per_participant = emg_dict["lambda"]

    single_correlation = False

    if len(set(correlation_m1)) == 1:
        x = gen_gaussian_copula(1000, correlation_m1[0], viz=False)
        x_unif = stats.norm.cdf(x)
        single_correlation = True
    else:
        copula_dict = {}

    rows = []
    # p x strat x block x trials x ide x mt

    # participant distributions:
    xp_data = {}
    for np, (beta, sigma, _lambda) in tqdm(
        enumerate(
            zip(beta_per_participant, sigma_per_participant, lambda_emg_per_participant)
        )
    ):
        # sample parameters
        p_data = {}
        for ns, (strat, correlation, mu, std) in tqdm(
            enumerate(zip(strategies, correlation_m1, loc_m1, scale_m1))
        ):
            if not single_correlation:
                if np == 0:
                    x = gen_gaussian_copula(1000, correlation, viz=False)
                    x_unif = stats.norm.cdf(x)
                    copula_dict[ns] = x
                else:
                    x = copula_dict[ns]

            X, Y = [], []

            for nb in tqdm(range(nblocks)):
                x, y = gen_emg_data_gaussian_copula_strategy(
                    x_unif, beta, sigma, _lambda, mu, std, ntrials
                )
                X.append(x)
                Y.append(y)
                for nt, (_x, _y) in enumerate(zip(x, y)):
                    rows.append([np, ns, nb, nt, _x, _y])

            p_data[strat] = (X, Y)
        xp_data[np] = p_data

    df = polars.DataFrame(
        rows, schema=["participants", "strategy", "block", "trial", "id", "mt"]
    )

    if viz:
        h = seaborn.jointplot(x=df["id"], y=df["mt"], kind="kde")
        h = seaborn.jointplot(
            x=df["id"], y=df["mt"], kind="scatter", joint_kws={"s": 4}
        )

    return xp_data, df


if __name__ == "__main__":

    def viz_df(df, ax=None):
        return seaborn.scatterplot(data=df, x="id", y="mt", hue="strategy", ax=ax)

    N_participant = 12

    # hierarchical model on the EMG
    beta_rv = stats.multivariate_normal(
        mean=numpy.array([0.07823552, 0.20838737]),
        cov=numpy.array([[0.00323031, -0.00136086], [-0.00136086, 0.0010003]]),
    )
    beta_per_participant = beta_rv.rvs(N_participant)
    beta_per_participant = [numpy.array([0.1, 0.2]) for i in range(N_participant)]

    sigma_rv = stats.uniform(loc=0.04, scale=0.05)
    sigma_per_participant = sigma_rv.rvs(N_participant)
    sigma_per_participant = [0.1 for i in range(N_participant)]

    lambda_emg_rv = stats.norm(loc=0.1, scale=0.02)
    lambda_emg_per_participant = [(0.05, v) for v in lambda_emg_rv.rvs(N_participant)]
    lambda_emg_per_participant = [
        numpy.array([0.05, 0.1]) for i in range(N_participant)
    ]

    # Generate pointing data without explicitly controlling for the correlation between IDe and mean MT
    strategies = [-1, -0.5, 0, 0.5, 1]
    mu = [4.73 + strat * 2.18 for strat in strategies]
    std = [1.03 + 0.28 * strat for strat in strategies]
    gaussian_ide = {"label": strategies, "mu": mu, "std": std}

    xp_data_no_control, df_no_control = (
        gen_emg_data_strategy_no_correlation_control_gaussian_ide(
            beta_per_participant,
            sigma_per_participant,
            lambda_emg_per_participant,
            gaussian_ide,
            nblocks=5,
            ntrials=15,
        )
    )

    # Generate pointing data correcting for correlation between IDe and mean MT
    strategies = [-1, -0.5, 0, 0.5, 1]
    mu = [
        numpy.array([4.73 + strat * 2.18, 1.28 + 0.65 * strat]) for strat in strategies
    ]
    crosscovariance = [
        (1.03 + 0.28 * strat) * (0.37 + 0.1 * strat) * 0.36 for strat in strategies
    ]
    covariance = [
        numpy.array(
            [
                [(1.03 + 0.28 * strat) ** 2, cc],
                [cc, (0.37 + 0.1 * strat) ** 2],
            ]
        )
        for strat, cc in zip(strategies, crosscovariance)
    ]
    gaussian_joint_dict = {"label": strategies, "mu": mu, "covariance": covariance}
    xp_data_correct, df_correct = (
        gen_emg_data_strategy_correlation_control_gaussian_ide(
            beta_per_participant,
            sigma_per_participant,
            lambda_emg_per_participant,
            gaussian_joint_dict,
            nblocks=5,
            ntrials=15,
        )
    )

    # Generate pointing data with Gaussian copulas
    strategies = [-1, -0.5, 0, 0.5, 1]
    loc_m1 = [4.73 + strat * 2.18 for strat in strategies]
    scale_m1 = [1.03 + 0.28 * strat for strat in strategies]
    correlation = [0.36 for strat in strategies]
    # mu_emg = numpy.array([0.1, 0.2])
    # lambda_emg = numpy.array([0.05, 0.1])
    copula_dict = {
        "label": strategies,
        "correlation": correlation,
        "mu": loc_m1,
        "scale": scale_m1,
    }

    # beta = numpy.array([0.1, 0.2])
    # _lambda = numpy.array([0.05, 0.1])
    # sigma = 0.1
    emg_dict = {
        "beta": beta_per_participant,
        "sigma": sigma_per_participant,
        "lambda": lambda_emg_per_participant,
    }

    xp_data_copula, df_copula = gen_emg_data_gaussian_copula(
        copula_dict, emg_dict, nblocks=5, ntrials=15, viz=False
    )

fig, axs = plt.subplots(1, 3)
viz_df(df_no_control, ax=axs[0])
viz_df(df_correct, ax=axs[1])
viz_df(df_copula, ax=axs[2])
    plt.show()
