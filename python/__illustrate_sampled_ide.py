### This file to show that we can also sample IDe from some learned distribution, rather than using empirical IDe values.


# from simulate import gen_t_copula, gen_emg, gen_emg_control
from gen_joint_mtide.core.simulate import gen_t_copula, gen_emg, gen_emg_control
import polars
import matplotlib.pyplot as plt
import seaborn
import numpy
from scipy.stats import gamma, pearsonr, spearmanr, kendalltau, multivariate_normal
import statsmodels.api as sm
from emg_arbitrary_variance import compute_emg_regression_linear_expo_mean

plt.style.use("fivethirtyeight")

colors = [
    "#008fd5",
    "#fc4f30",
    "#e5ae38",
    "#6d904f",
    "#8b8b8b",
    "#810f7c",
    "#ff5733",
    "#33ff57",
    "#3357ff",
    "#ff33aa",
    "#aaff33",
    "#33ffaa",
    "#ff3333",
    "#33ff33",
    "#3333ff",
    "#ff9933",
    "#9933ff",
    "#33ff99",
    "#99ff33",
    "#33ccff",
    "#ff33cc",
]


def remove_data_points_below_thres(df, thresh=0.1):
    return df.filter(polars.col("MT") >= thresh)


def compute_associations(df):
    return [
        pearsonr(df["IDe"], df["MT"]).statistic,
        spearmanr(df["IDe"], df["MT"]).statistic,
        kendalltau(df["IDe"], df["MT"]).statistic,
    ]


def compute_ols_means(df):
    try:
        df_mean = df["IDe", "MT"].groupby("IDe").mean()
    except AttributeError:
        df_mean = df["IDe", "MT"].group_by("IDe").mean()
    X = sm.add_constant(df_mean["IDe"])
    model = sm.OLS(numpy.asarray(df_mean["MT"]), X)
    results = model.fit()
    r = numpy.sqrt(results.rsquared)
    coefs = results.params
    return coefs, r, results


def compute_emg_regression(df):
    x, y = df["IDe"], df["MT"]
    return compute_emg_regression_linear_expo_mean(x, y)


def compute_ISO_throughput(df):
    try:
        df_mean = df.groupby(["IDe"]).mean()
    except AttributeError:
        df_mean = df.group_by(["IDe"]).mean()
    df_mean = df_mean.with_columns((df_mean["IDe"] / df_mean["MT"]).alias("throughput"))
    ISO_throughput = df_mean["throughput"].mean()
    return ISO_throughput


if __name__ == "__main__":

    df = polars.read_csv("fitts_csv_GOP.csv", has_header=True, ignore_errors=True)
    # remove P9 data inconsistent with other Ps
    df = df.filter(polars.col("Participant") != 9)

    participants = df["Participant"].unique()
    strategies = df["strategy"].unique()
    repetitions = df["repetition"].unique()

    import pandas

    df_pandas = df.to_pandas()
    ##################################
    fig, axs = plt.subplots(2, 4)

    #####  Raw dataset
    seaborn.scatterplot(df_pandas, x="IDe", y="MT", ax=axs[0, 0])
    seaborn.scatterplot(df_pandas, x="IDe", y="MT", hue="strategy", ax=axs[1, 0])

    #### Fitts' law -- mean per Participant and IDe
    mean_df = df_pandas.groupby(["Participant", "IDe"]).mean()
    seaborn.scatterplot(mean_df, x="IDe", y="MT", ax=axs[0, 1])
    seaborn.scatterplot(mean_df, x="IDe", y="MT", hue="strategy", ax=axs[1, 1])

    plt.ion()
    plt.show()

    #### ==== Estimation phase ====

    vec_original = compute_associations(df)
    original_ols_fit = compute_ols_means(df)
    original_emg_fit = compute_emg_regression(df)
    tp_original = compute_ISO_throughput(df)

    ### joint mmt, ide, without strategy explitcly ====== model 3

    from eval_data import gaussian_fit

    try:
        df_mean = df["IDe", "MT"].groupby("IDe").mean()
    except AttributeError:
        df_mean = df["IDe", "MT"].group_by("IDe").mean()
    gaussian_fit(df_mean["IDe"], df_mean["MT"], ax=axs[0, 2])

    ### joint mmt, ide per strategy ===== model 3
    means_strat, covs_strat = [], []
    for strat in strategies:
        tmp_df = df.filter(polars.col("strategy") == strat)
        try:
            tmp_df_mean = tmp_df["IDe", "MT"].groupby("IDe").mean()
        except AttributeError:
            tmp_df_mean = tmp_df["IDe", "MT"].group_by("IDe").mean()

        gaussian_fit(
            tmp_df_mean["IDe"], tmp_df_mean["MT"], ax=axs[1, 2], label=str(strat)
        )

    plt.ion()
    plt.show()
    exit()
    SEED = 1234

    #### ==== Gen phase =====

    x_3, y_3 = gen_emg_control(
        beta,
        sigma,
        lambda_emg,
        mean,
        cov,
        block_levels=block_levels,
        ntrials=ntrials,
        rng=None,
        seed=SEED,
    )

    ## ==== Eval phase ====
    df_gen_copula = polars.DataFrame({"IDe": x_1, "MT": y_1})
    df_gen_copula = remove_data_points_below_thres(df_gen_copula, thresh=0.2)
    x_1, y_1 = df_gen_copula["IDe"], df_gen_copula["MT"]
    vec_gen_copula = compute_associations(df_gen_copula)
    gen_ols_fit_copula = compute_ols_means(df_gen_copula)
    gen_emg_git_copula = compute_emg_regression(df_gen_copula)
    tp_gen_copula = compute_ISO_throughput(df_gen_copula)

    df_gen_emg_1 = polars.DataFrame({"IDe": x_2, "MT": y_2})
    df_gen_emg_1 = remove_data_points_below_thres(df_gen_emg_1, thresh=0.2)
    x_2, y_2 = df_gen_emg_1["IDe"], df_gen_emg_1["MT"]
    vec_gen_emg_1 = compute_associations(df_gen_emg_1)
    gen_ols_fit_emg_1 = compute_ols_means(df_gen_emg_1)
    gen_emg_git_emg_1 = compute_emg_regression(df_gen_emg_1)
    tp_gen_emg_1 = compute_ISO_throughput(df_gen_emg_1)

    df_gen_emg_2 = polars.DataFrame({"IDe": x_3, "MT": y_3})
    df_gen_emg_2 = remove_data_points_below_thres(df_gen_emg_2, thresh=0.2)
    x_3, y_3 = df_gen_emg_2["IDe"], df_gen_emg_2["MT"]
    vec_gen_emg_2 = compute_associations(df_gen_emg_2)
    gen_ols_fit_emg_2 = compute_ols_means(df_gen_emg_2)
    gen_emg_git_emg_2 = compute_emg_regression(df_gen_emg_2)
    tp_gen_emg_2 = compute_ISO_throughput(df_gen_emg_2)

    ### ==== plots ====
    fig, axs = plt.subplots(1, 4)
    seaborn.scatterplot(df, x="IDe", y="MT", ax=axs[0])
    seaborn.scatterplot(x=x_1, y=y_1, ax=axs[1])
    seaborn.scatterplot(x=x_2, y=y_2, ax=axs[2])
    seaborn.scatterplot(x=x_3, y=y_3, ax=axs[3])

    # ylim
    axs0_lim = axs[0].get_ylim()
    axs1_lim = axs[1].get_ylim()
    axs2_lim = axs[2].get_ylim()
    axs3_lim = axs[3].get_ylim()

    axs_lim = (
        numpy.min((axs0_lim[0], axs1_lim[0], axs2_lim[0], axs3_lim[0])),
        numpy.max((axs0_lim[1], axs1_lim[1], axs2_lim[1], axs3_lim[1])),
    )

    axs[0].set_ylim(axs_lim)
    axs[1].set_ylim(axs_lim)
    axs[2].set_ylim(axs_lim)
    axs[3].set_ylim(axs_lim)

    # xlim
    axs0_lim = axs[0].get_xlim()
    axs1_lim = axs[1].get_xlim()
    axs2_lim = axs[2].get_xlim()
    axs3_lim = axs[3].get_xlim()

    axs_lim = (
        numpy.min((axs0_lim[0], axs0_lim[0], axs2_lim[0], axs3_lim[0])),
        numpy.max((axs0_lim[1], axs1_lim[1], axs2_lim[1], axs3_lim[1])),
    )

    axs[0].set_xlim(axs_lim)
    axs[1].set_xlim(axs_lim)
    axs[2].set_xlim(axs_lim)
    axs[3].set_xlim(axs_lim)

    # labels
    axs[0].set_xlabel("IDe (bit)")
    axs[1].set_xlabel("IDe (bit)")
    axs[2].set_xlabel("IDe (bit)")
    axs[3].set_xlabel("IDe (bit)")
    axs[0].set_ylabel("MT (s)")
    axs[1].set_ylabel("MT (s)")
    axs[2].set_ylabel("MT (s)")
    axs[3].set_ylabel("MT (s)")

    # plot ols and emg
    x_vals = [x_1.min(), x_1.max()]
    axs[0].plot(
        x_vals,
        [original_ols_fit[0][0] + original_ols_fit[0][1] * _x for _x in x_vals],
        "-",
        color=colors[1],
        label=f"OLS fit: MT = {original_ols_fit[0][0]:.2f} + {original_ols_fit[0][1]:.2f}IDe",
    )
    axs[1].plot(
        x_vals,
        [gen_ols_fit_copula[0][0] + gen_ols_fit_copula[0][1] * _x for _x in x_vals],
        "-",
        color=colors[1],
        label=f"OLS fit: MT = {gen_ols_fit_copula[0][0]:.2f} + {gen_ols_fit_copula[0][1]:.2f}IDe",
    )
    axs[2].plot(
        x_vals,
        [gen_ols_fit_emg_1[0][0] + gen_ols_fit_emg_1[0][1] * _x for _x in x_vals],
        "-",
        color=colors[1],
        label=f"OLS fit: MT = {gen_ols_fit_emg_1[0][0]:.2f} + {gen_ols_fit_emg_1[0][1]:.2f}IDe",
    )
    axs[3].plot(
        x_vals,
        [gen_ols_fit_emg_2[0][0] + gen_ols_fit_emg_2[0][1] * _x for _x in x_vals],
        "-",
        color=colors[1],
        label=f"OLS fit: MT = {gen_ols_fit_emg_2[0][0]:.2f} + {gen_ols_fit_emg_2[0][1]:.2f}IDe",
    )

    axs[0].plot(
        x_vals,
        [original_emg_fit[0][0] + original_emg_fit[0][1] * _x for _x in x_vals],
        "-",
        color=colors[2],
        label=f"EMG fit: MT = {original_emg_fit[0][0]:.2f} + {original_emg_fit[0][1]:.2f}IDe",
    )
    axs[1].plot(
        x_vals,
        [gen_emg_git_copula[0][0] + gen_emg_git_copula[0][1] * _x for _x in x_vals],
        "-",
        color=colors[2],
        label=f"EMG fit: MT = {gen_emg_git_copula[0][0]:.2f} + {gen_emg_git_copula[0][1]:.2f}IDe",
    )
    axs[2].plot(
        x_vals,
        [gen_emg_git_emg_1[0][0] + gen_emg_git_emg_1[0][1] * _x for _x in x_vals],
        "-",
        color=colors[2],
        label=f"EMG fit: MT = {gen_emg_git_emg_1[0][0]:.2f} + {gen_emg_git_emg_1[0][1]:.2f}IDe",
    )
    axs[3].plot(
        x_vals,
        [gen_emg_git_emg_2[0][0] + gen_emg_git_emg_2[0][1] * _x for _x in x_vals],
        "-",
        color=colors[2],
        label=f"EMG fit: MT = {gen_emg_git_emg_2[0][0]:.2f} + {gen_emg_git_emg_2[0][1]:.2f}IDe",
    )

    # titles, legend
    axs[0].set_title(
        "Ground Truth dataset\n"
        + rf"r = {vec_original[0]:.2f}, $\rho$ = {vec_original[1]:.2f}, $\tau$ = {vec_original[2]:.2f},"
        + "\n"
        + rf"$\overline{{r}}$ = {original_ols_fit[1]:.2f}, ISO-TP = {tp_original:.2f}"
    )
    axs[1].set_title(
        "Model 1 (copulas)\n"
        + rf"r = {vec_gen_copula[0]:.2f}, $\rho$ = {vec_gen_copula[1]:.2f}, $\tau$ = {vec_gen_copula[2]:.2f},"
        + "\n"
        + rf"$\overline{{r}}$ = { gen_ols_fit_copula[1]:.2f}, ISO-TP = {tp_gen_copula:.2f}"
    )
    axs[2].set_title(
        "Model 2 (EMG)\n"
        + rf"r = {vec_gen_emg_1[0]:.2f}, $\rho$ = {vec_gen_emg_1[1]:.2f}, $\tau$ = {vec_gen_emg_1[2]:.2f},"
        + "\n"
        + rf"$\overline{{r}}$ = { gen_ols_fit_emg_1 [1]:.2f}, ISO-TP = {tp_gen_emg_1:.2f}"
    )
    axs[3].set_title(
        "Model 3 (EMG with control)\n"
        + rf"r = {vec_gen_emg_2[0]:.2f}, $\rho$ = {vec_gen_emg_2[1]:.2f}, $\tau$ = {vec_gen_emg_2[2]:.2f},"
        + "\n"
        + rf"$\overline{{r}}$ = { gen_ols_fit_emg_2 [1]:.2f}, ISO-TP = {tp_gen_emg_2:.2f}"
    )
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    axs[3].legend()

    plt.ion()
    plt.tight_layout()
    plt.show()
    # fig.savefig("img/method_consistency.pdf")
    # fig.savefig("supp_source/method_consistency_seed777.pdf")
