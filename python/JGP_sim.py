import pickle
import numpy
from emgregs import sim_emg_reg_heterosked
import matplotlib.pyplot as plt
import statsmodels.api as sm

with open("JGP_params.pkl", "rb") as _file:
    params = pickle.load(_file)


def simulate_fitts_full_factorial(
    params,
    N_participants=16,
    trials_per_block=40,
    D_factor_level=[256, 512, 1024, 1408],
    W_factor_level=[64, 96, 128],
):

    def sample_participant(params):
        beta = numpy.random.multivariate_normal(**params["beta"]["params"])
        sigma = numpy.random.uniform(
            low=params["sigma"]["params"]["min"], high=params["sigma"]["params"]["max"]
        )
        expo0 = numpy.random.uniform(
            low=params["expo"]["params"]["min"][0],
            high=params["expo"]["params"]["max"][0],
        )
        expo1 = numpy.random.uniform(
            low=params["expo"]["params"]["min"][1],
            high=params["expo"]["params"]["max"][1],
        )
        return beta, sigma, [expo0, expo1]

    output = {}

    for p in range(N_participants):
        id_list = []
        for D in D_factor_level:
            for W in W_factor_level:
                id_list.extend([numpy.log2(1 + D / W) for n in range(trials_per_block)])

        beta, sigma, expo = sample_participant(params)
        X = sm.add_constant(id_list)
        sim_data = sim_emg_reg_heterosked(
            X=X, beta=beta, sigma=sigma, expo_scale=expo, xmin=0.01
        )

        output[p] = sim_data
    return output


if __name__ == "__main__":
    sim_data = simulate_fitts_full_factorial(
        params,
        N_participants=16,
        trials_per_block=40,
        D_factor_level=[256, 512, 1024, 1408],
        W_factor_level=[64, 96, 128],
    )

    fig, axs = plt.subplots(4, 4)
    for p in range(16):
        axs[p // 4, p % 4].plot(sim_data[p]["X"], sim_data[p]["Y"], ".")

    plt.tight_layout()
    plt.ion()
    plt.show()
