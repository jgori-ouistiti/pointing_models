import scipy.stats as stats
import numpy
import polars
import seaborn
import matplotlib.pyplot as plt
from tqdm import tqdm

import rpy2.robjects as robjects
from rpy2.robjects.vectors import FloatVector
from rpy2.rlike.container import TaggedList

robjects.r["source"]("gen_t_copula.R")
gen_copula_fun = robjects.globalenv["gen_block_t"]


def python_params_to_named_list_R(obj):
    params = obj["params"]
    params_list = TaggedList(list(params.values()), tags=list(params.keys()))
    obj_list = TaggedList([obj["distribution"], params_list], tags=list(obj.keys()))
    return obj_list


def correct_beta_lambda(ide, mt, beta, lambda_emg):
    lambda_corrected = (
        lambda_emg[0],
        numpy.maximum(
            0,
            (mt - beta[0] - beta[1] * ide - lambda_emg[0]) / ide,
        ),
    )
    beta_corrected = beta

    # lambda_corrected = lambda_emg
    # beta_corrected = mt - lambda_emg[0] - lambda_emg[1] * ide - beta[1] * ide
    return beta_corrected, lambda_corrected


def gen_emg(
    beta, sigma, lambda_emg, block_levels=None, ntrials=50, rng=None, seed=None
):
    ide = block_levels
    if ide is None:
        ide = numpy.linspace(0.1, 0.9, 10)
    elif isinstance(ide, int):
        ide = list(rng.random(ide) * 8)
    else:
        pass

    X = numpy.full((ntrials, len(ide)), fill_value=ide)
    loc = beta[0] + beta[1] * X
    scale = sigma
    expo_mean = lambda_emg[0] + lambda_emg[1] * X
    X = X.ravel()
    loc = loc.ravel()
    expo_mean = expo_mean.ravel()
    K = expo_mean / scale
    y = stats.exponnorm(K, loc=loc, scale=scale).rvs(random_state=seed)
    return X, y


def gen_emg_control(
    beta,
    sigma,
    lambda_emg,
    mvg_mu,
    mvg_cov,
    block_levels=None,
    ntrials=50,
    rng=None,
    seed=None,
):
    if rng is None:
        rng = numpy.random.default_rng(seed=seed)

    rng = rng.spawn(1)[0]
    if seed is None:
        seed = int(numpy.floor(rng.random(1)[0] * 999))

    ide = block_levels

    if ide is not None:
        if isinstance(ide, int):
            ide = list(rng.random(ide) * 8)
        else:
            ide = numpy.asarray(ide)

        mu_mt = mvg_mu[1] + mvg_cov[0, 1] / mvg_cov[0, 0] * (ide - mvg_mu[0])
        var = mvg_cov[1, 1] - mvg_cov[0, 1] ** 2 / mvg_cov[0, 0]
        y = stats.norm(loc=mu_mt, scale=var).rvs()
    else:
        x, y = stats.multivariate_normal(mean=mvg_mu, cov=mvg_cov).rvs()

    beta, lambda_emg = correct_beta_lambda(x, y, beta, lambda_emg)

    return gen_emg(
        beta, sigma, lambda_emg, block_levels=[x], ntrials=ntrials, seed=seed
    )


def gen_t_copula(
    rho1,
    df,
    id_params,
    mt_params,  # for the marginals, pass things that make sense to R
    trials=15,
    block_levels=None,
    cdf_block=False,
    rng=None,
    seed=None,
):

    if rng is None:
        rng = numpy.random.default_rng(seed=seed)

    rng = rng.spawn(1)[0]
    if seed is None:
        seed = int(numpy.floor(rng.random(1)[0] * 999))

    if block_levels is None:
        block_levels = FloatVector(list(numpy.linspace(0.1, 0.9, 10)))
    elif isinstance(block_levels, int):
        block_levels = FloatVector(list(rng.random(block_levels)))
    else:
        if cdf_block:
            if id_params["distribution"] == "gamma":
                cdf = [
                    getattr(stats, id_params["distribution"]).cdf(
                        b,
                        id_params["params"]["shape"],
                        scale=(1 / id_params["params"]["rate"]),
                    )
                    for b in block_levels
                ]
            else:
                raise NotImplementedError
        else:
            cdf = block_levels
        block_levels = FloatVector(cdf)

    _array = numpy.array(
        gen_copula_fun(
            float(rho1),
            float(df),
            python_params_to_named_list_R(id_params),
            python_params_to_named_list_R(mt_params),
            FloatVector(block_levels),
            int(trials),
            int(seed),
        )
    )
    return _array[:, 0], _array[:, 1]


# def gen_copula_strat(
#     lambda_per_participant,
#     strategies,
#     copula_theta,
#     gaussian_ide_marginal,
#     repetitions,
#     block_levels,
#     trials,
# ):

#     rows = []
#     xp_data = {}
#     for np, (beta, sigma, _lambda) in enumerate(
#         zip(beta_per_participant, sigma_per_participant, lambda_per_participant)
#     ):
#         data = {}
#         for nstrat, strat in enumerate(strategies):
#             strat_data = ()
#             for repetition in range(repetitions):
#                 x, y = _gen_gumbel_copula(
#                     copula_theta,
#                     gaussian_ide_marginal["mu"][nstrat],
#                     gaussian_ide_marginal["sigma"][nstrat],
#                     beta,
#                     sigma,
#                     _lambda,
#                     trials=trials,
#                     block_levels=block_levels,
#                     rng=None,
#                     seed=None,
#                 )
#                 for nt, (_x, _y) in enumerate(zip(x, y)):
#                     rows.append([np, nstrat, repetition, nt, _x, _y])
#                 strat_data = strat_data + ((x, y),)

#             data[strat] = strat_data

#         xp_data[np] = data
#     df = polars.DataFrame(
#         rows, schema=["participants", "strategy", "block", "trial", "id", "mt"]
#     )
#     return xp_data, df


def gen_emg_data_strategy_no_correlation_control_gaussian_ide(
    beta_per_participant,
    sigma_per_participant,
    lambda_emg_per_participant,
    gaussian_ide_dict,
    nblocks=5,
    ntrials=15,
    rng=None,
    seed=None,
):

    if rng is None:
        rng = numpy.random.default_rng(seed=seed)

    rngs = rng.spawn(2)

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
            ide = ide_distrib.rvs(nblocks, random_state=rngs[0])

            X, Y = [], []

            for nb, block in enumerate(ide):
                x, y = gen_emg(
                    beta, sigma, _lambda, ide=block, ntrials=ntrials, rng=rngs[1]
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
    return xp_data, df


def gen_emg_data_strategy_correlation_control_gaussian_ide(
    beta_per_participant,
    sigma_per_participant,
    lambda_emg_per_participant,
    gaussian_joint_dict,
    nblocks=5,
    ntrials=15,
    rng=None,
    seed=None,
):

    if rng is None:
        rng = numpy.random.default_rng(seed=seed)

    rngs = rng.spawn(2)

    strategy = gaussian_joint_dict["label"]
    MU = gaussian_joint_dict["mu"]
    covariance = gaussian_joint_dict["covariance"]

    rows = []
    # p x strat x block x trials x ide x mt

    # participant distributions:
    jg_data = []
    xp_data = {}
    for np, (beta, sigma, _lambda) in enumerate(
        zip(beta_per_participant, sigma_per_participant, lambda_emg_per_participant)
    ):
        # sample parameters
        p_data = {}
        for ns, (strat, mu, cov) in enumerate(zip(strategies, MU, covariance)):
            bivariate_gaussian = stats.multivariate_normal(mean=mu, cov=cov)
            ide, mt = bivariate_gaussian.rvs(nblocks, random_state=rngs[0]).T
            jg_data.append((np, ns, ide, mt))
            X, Y = [], []

            for nb, (_id, _mt) in enumerate(zip(ide, mt)):
                # beta_corrected, lambda_corrected = correct_beta_lambda(
                #     _id, _mt, beta, _lambda
                # )
                # print("correction")
                # print(f"{beta} --> {beta_corrected}")
                # print(f"{_lambda} --> {lambda_corrected}")
                lambda_emg = (
                    _lambda[0],
                    # max(_lambda[1], (_mt - beta[0] - beta[1] * _id) / _id),
                    max(0, (_mt - beta[0] - beta[1] * _id) / _id),
                )  # correct lambda
                x, y = gen_emg(
                    beta, sigma, lambda_emg, ide=_id, ntrials=ntrials, rng=rngs[1]
                )
                # corrected_beta = (
                #     (_mt - beta[1] * _id - _lambda[0] - _lambda[1] * _id),
                #     beta[1],
                # )
                # x, y = gen_emg(
                #     beta_corrected,
                #     sigma,
                #     lambda_corrected,
                #     ide=_id,
                #     ntrials=ntrials,
                #     rng=rngs[1],
                # )
                X.append(x)
                Y.append(y)
                for nt, (_x, _y) in enumerate(zip(x, y)):
                    rows.append([np, ns, nb, nt, _x, _y])

            p_data[strat] = (X, Y)
        xp_data[np] = p_data

    df = polars.DataFrame(
        rows, schema=["participants", "strategy", "block", "trial", "id", "mt"]
    )
    return xp_data, df, jg_data


if __name__ == "__main__":

    # id_params = {"distribution": "norm", "params": dict(mean=1, sd=1)}

    # tagged_list = python_params_to_named_list_R(id_params)

    def viz_df(df, ax=None):
        return seaborn.scatterplot(data=df, x="id", y="mt", hue="strategy", ax=ax)

    N_participant = 12
    seed = 777
    rng = numpy.random.default_rng(seed=seed)
    rngs = rng.spawn(6)

    # hierarchical model on the EMG
    beta_rv = stats.multivariate_normal(
        mean=numpy.array([0.07823552, 0.20838737]),
        cov=numpy.array([[0.00323031, -0.00136086], [-0.00136086, 0.0010003]]),
    )
    beta_per_participant = beta_rv.rvs(N_participant, random_state=rngs[0])

    sigma_rv = stats.uniform(loc=0.2, scale=0.05)

    sigma_per_participant = sigma_rv.rvs(N_participant, random_state=rngs[1])

    lambda_emg_rv = stats.norm(loc=0.1, scale=0.02)
    lambda_emg_per_participant = [
        (0.05, v) for v in lambda_emg_rv.rvs(N_participant, random_state=rngs[2])
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
            rng=rngs[3],
        )
    )

    # Generate pointing data correcting for correlation between IDe and mean MT
    strategies = [-1, -0.5, 0, 0.5, 1]
    mu_bv = [
        numpy.array([4.73 + strat * 2.18, 1.28 + 0.65 * strat]) for strat in strategies
    ]
    covariance = [
        (1.03 + 0.28 * strat) * (0.37 + 0.1 * strat) * 0.36 for strat in strategies
    ]
    covariance = [
        numpy.array(
            [
                [(1.03 + 0.28 * strat) ** 2, cc],
                [cc, (0.37 + 0.1 * strat) ** 2],
            ]
        )
        for strat, cc in zip(strategies, covariance)
    ]
    gaussian_joint_dict = {"label": strategies, "mu": mu_bv, "covariance": covariance}
    xp_data_correct, df_correct, jg_data = (
        gen_emg_data_strategy_correlation_control_gaussian_ide(
            beta_per_participant,
            sigma_per_participant,
            lambda_emg_per_participant,
            gaussian_joint_dict,
            nblocks=5,
            ntrials=15,
            rng=rngs[4],
        )
    )

    rho1 = 0.75
    df = 3.75
    id_params = {"distribution": "norm", "params": dict(mean=1, sd=1)}
    mt_params = {"distribution": "emg", "params": {"mu": 1, "sigma": 1, "lambda": 1}}
    sim_data = gen_t_copula(
        rho1,
        df,
        id_params,
        mt_params,
        trials=15,
        block_levels=None,
        rng=None,
        seed=None,
    )
    # Check if sample data is well correlated
    # corr = numpy.zeros((N_participant, 5))
    # for np, ns, ide, mt in jg_data:
    #     mean, cov = stats.multivariate_normal.fit(
    #         numpy.array(object=[numpy.asarray(ide), numpy.asarray(mt)]).T
    #     )
    #     rho = cov[0, 1] / numpy.sqrt(cov[0, 0] * cov[1, 1])
    #     corr[np, ns] = rho

    # mono_beta_per_participant = [u[0] + 4 * u[1] for u in beta_per_participant]
    # lambda_per_participant = [u[0] + 4 * u[1] for u in lambda_emg_per_participant]
    # copula_theta = 2.3
    # gaussian_ide_marginal = dict(mu=mu, sigma=std)
    # repetitions = 5
    # block_levels = 5
    # trials = 15
    # copula_data, copula_df = gen_copula_strat(
    #     mono_beta_per_participant,
    #     sigma_per_participant,
    #     lambda_per_participant,
    #     strategies,
    #     copula_theta,
    #     gaussian_ide_marginal,
    #     repetitions,
    #     block_levels,
    #     trials,
    # )

    from eval_data import *
    import seaborn

    fig, axs = plt.subplots(1, 3)
    viz_df(df_no_control, ax=axs[0])
    viz_df(df_correct, ax=axs[1])
    # viz_df(copula_df, ax=axs[2])
    # viz_df(df_copula_cutoff, ax=axs[2])
    handles = evaluate_data_mean_mvgauss_df(df_no_control)
    handles = evaluate_data_mean_mvgauss_df(df=df_correct)
    # handles = evaluate_data_mean_mvgauss_df(df=copula_df)
    # handles = evaluate_data_mean_mvgauss_df(df=df_copula_cutoff)
    plt.ion()
    plt.show()
