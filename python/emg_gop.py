import polars
import numpy
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import seaborn
import statsmodels.api as sm
from scipy.stats import multivariate_normal, pearsonr, kendalltau, spearmanr
import pickle
from emg_arbitrary_variance import (
    compute_emg_regression_linear_expo_mean,
    compute_gaussian_regression_linear_expo_mean,
)


plt.style.use("fivethirtyeight")


### Controls
FIG_OUTPUT = False

### Unpacking data


df = polars.read_csv("fitts_csv_GOP.csv", has_header=True, ignore_errors=True)
# remove P9 strange data
df = df.filter(polars.col("Participant") != 9)

participants = df["Participant"].unique()
strategies = df["strategy"].unique()
repetitions = df["repetition"].unique()

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


participants = df["Participant"].unique()
label_mapping = {
    "1": "Speed Emphasis",
    "2": "Speed",
    "3": "Balanced",
    "4": "Accuracy",
    "5": "Accuracy Emphasis",
}

fontdict = {"fontsize": 14}
emg_rows = []


for participant in participants:
    participant_data = {}
    df_part = df.filter(polars.col("Participant") == participant)
    strat = df_part["strategy"].unique()
    strat_dict = {"1": True, "2": True, "3": True, "4": True, "5": True}

    repetitions = df_part["repetition"].unique()
    for s in strat:
        df_block = df_part.filter(polars.col("strategy") == s)
        x, y = df_block["IDe"], df_block["MT"]
        try:
            params, fit = compute_emg_regression_linear_expo_mean(x, y)
            params_gauss, fit_gauss = compute_gaussian_regression_linear_expo_mean(x, y)
            X = sm.add_constant(numpy.asarray(x))
            ols_fit = sm.OLS(numpy.asarray(y), exog=X).fit()
            ols_params = ols_fit.params
            ols_aic = ols_fit.aic
            gauss_aic = 2 * len(fit_gauss.x) + 2 * fit_gauss.fun  # fit.fun returns -ll
            emg_aic = 2 * len(fit.x) + 2 * fit.fun
            R = numpy.exp((emg_aic - ols_aic) / 2)

        except RuntimeError:
            params = [numpy.nan for i in range(5)]

        emg_rows.append(
            [
                participant,
                s,
                *params,
                *ols_fit.params,
                numpy.sqrt(ols_fit.scale),
                *params_gauss,
                emg_aic,
                ols_aic,
                gauss_aic,
            ]
        )
        fig, ax = plt.subplots(1, 1)
        ax.plot(
            df_block["IDe"],
            df_block["MT"],
            marker="o",
            linestyle="",
        )
        vec = [
            pearsonr(x, y).statistic,
            spearmanr(x, y).statistic,
            kendalltau(x, y).statistic,
        ]
        ax.set_title(
            rf"P{participant},S{s}: r = {vec[0]:.2f}, $\rho$ = {vec[1]:.2f}, $\tau$ = {vec[2]:.2f}",
            fontdict=fontdict,
        )
        vec_x = [numpy.min(numpy.asarray(x)), numpy.max(numpy.asarray(x))]
        ax.plot(vec_x, [params[0] + params[1] * v for v in vec_x], "-", label=f"EMG")
        ax.plot(
            vec_x,
            [params_gauss[0] + params_gauss[1] * v for v in vec_x],
            "-",
            label=f"Gauss QV R = {numpy.exp((emg_aic - gauss_aic) / 2):.2e}",
        )
        ax.plot(
            vec_x,
            [ols_params[0] + ols_params[1] * v for v in vec_x],
            "-",
            label=f"LR  R = {numpy.exp((emg_aic - ols_aic) / 2):.2e}",
        )
        ax.set_xlabel("IDe (bit)")
        ax.set_ylabel("MT (s)")
        ax.legend()
        # plt.ion()
        # plt.show()
        fig.tight_layout()
        if FIG_OUTPUT:
            fig.savefig(f"supp_source/gop/pooled_repet/fitts_gop_{participant}_{s}.pdf")
        plt.close(fig)


import pandas

emg_pooled_repet = pandas.DataFrame(
    emg_rows,
    columns=[
        "Participant",
        "Strategy",
        "emg_beta_0",
        "emg_beta_1",
        "emg_sigma",
        "emg_lambda_0",
        "emg_lambda_1",
        "lr_beta_0",
        "lr_beta_1",
        "lr_sigma",
        "gauss_qv_beta_0",
        "gauss_qv_beta_1",
        "gauss_qv_lambda_0",
        "gauss_qv_lambda_1",
        "emg_aic",
        "lr_aic",
        "gauss_qv_aic",
    ],
)

emg_pooled_repet["Variance model"] = (
    emg_pooled_repet["gauss_qv_aic"] - emg_pooled_repet["lr_aic"]
)
emg_pooled_repet["Symmetry"] = (
    emg_pooled_repet["emg_aic"] - emg_pooled_repet["gauss_qv_aic"]
)
df_emg_aic = emg_pooled_repet[["Variance model", "Symmetry"]]
df_melted = df_emg_aic.melt()
fig, ax = plt.subplots(1, 1)
seaborn.violinplot(data=df_melted, x="variable", y="value", ax=ax)
ax.set_xlabel("Modality compared")
ax.set_ylabel("AIC differences")
# fig.savefig("img/violin_go_pooled_repet.pdf")
plt.ion()
plt.show()

fig, axs = plt.subplots(1, 5)
for ni, i in enumerate(
    ["emg_beta_0", "emg_beta_1", "emg_sigma", "emg_lambda_0", "emg_lambda_1"]
):
    seaborn.scatterplot(data=emg_pooled_repet, x="Strategy", y=i, ax=axs[ni])
plt.ion()
plt.show()

exit()


for participant in participants:
    participant_data = {}
    df_part = df.filter(polars.col("Participant") == participant)
    strat = df_part["strategy"].unique()
    strat_dict = {"1": True, "2": True, "3": True, "4": True, "5": True}

    repetitions = df_part["repetition"].unique()
    for r in repetitions:
        df_block = df_part.filter(polars.col("repetition") == r)
        x, y = df_block["IDe"], df_block["MT"]
        try:
            params, fit = compute_emg_regression_linear_expo_mean(x, y)
            params_gauss, fit_gauss = compute_gaussian_regression_linear_expo_mean(x, y)
            X = sm.add_constant(numpy.asarray(x))
            ols_fit = sm.OLS(numpy.asarray(y), exog=X).fit()
            ols_params = ols_fit.params
            ols_aic = ols_fit.aic
            gauss_aic = 2 * len(fit_gauss.x) + 2 * fit_gauss.fun  # fit.fun returns -ll
            emg_aic = 2 * len(fit.x) + 2 * fit.fun
            R = numpy.exp((emg_aic - ols_aic) / 2)

        except RuntimeError:
            params = [numpy.nan for i in range(5)]

        emg_rows.append(
            [
                participant,
                r,
                *params,
                *ols_fit.params,
                numpy.sqrt(ols_fit.scale),
                *params_gauss,
                emg_aic,
                ols_aic,
                gauss_aic,
            ]
        )
        fig, ax = plt.subplots(1, 1)
        ax.plot(
            df_block["IDe"],
            df_block["MT"],
            marker="o",
            linestyle="",
        )
        vec = [
            pearsonr(x, y).statistic,
            spearmanr(x, y).statistic,
            kendalltau(x, y).statistic,
        ]
        ax.set_title(
            rf"P{participant},I{r}: r = {vec[0]:.2f}, $\rho$ = {vec[1]:.2f}, $\tau$ = {vec[2]:.2f}",
            fontdict=fontdict,
        )
        vec_x = [numpy.min(numpy.asarray(x)), numpy.max(numpy.asarray(x))]
        ax.plot(vec_x, [params[0] + params[1] * v for v in vec_x], "-", label=f"EMG")
        ax.plot(
            vec_x,
            [params_gauss[0] + params_gauss[1] * v for v in vec_x],
            "-",
            label=f"Gauss QV R = {numpy.exp((emg_aic - gauss_aic) / 2):.2e}",
        )
        ax.plot(
            vec_x,
            [ols_params[0] + ols_params[1] * v for v in vec_x],
            "-",
            label=f"LR  R = {numpy.exp((emg_aic - ols_aic) / 2):.2e}",
        )
        ax.set_xlabel("IDe (bit)")
        ax.set_ylabel("MT (s)")
        ax.legend()
        # plt.ion()
        # plt.show()
        fig.tight_layout()
        if FIG_OUTPUT:
            fig.savefig(f"supp_source/gop/pooled_strat/fitts_gop_{participant}_{r}.pdf")
        plt.close(fig)


import pandas

emg_pooled_strat = pandas.DataFrame(
    emg_rows,
    columns=[
        "Participant",
        "Repetition",
        "emg_beta_0",
        "emg_beta_1",
        "emg_sigma",
        "emg_lambda_0",
        "emg_lambda_1",
        "lr_beta_0",
        "lr_beta_1",
        "lr_sigma",
        "gauss_qv_beta_0",
        "gauss_qv_beta_1",
        "gauss_qv_lambda_0",
        "gauss_qv_lambda_1",
        "emg_aic",
        "lr_aic",
        "gauss_qv_aic",
    ],
)

emg_pooled_strat["Variance model"] = (
    emg_pooled_strat["gauss_qv_aic"] - emg_pooled_strat["lr_aic"]
)
emg_pooled_strat["Symmetry"] = (
    emg_pooled_strat["emg_aic"] - emg_pooled_strat["gauss_qv_aic"]
)
df_emg_aic = emg_pooled_strat[["Variance model", "Symmetry"]]
df_melted = df_emg_aic.melt()
fig, ax = plt.subplots(1, 1)
seaborn.violinplot(data=df_melted, x="variable", y="value", ax=ax)
ax.set_xlabel("Modality compared")
ax.set_ylabel("AIC differences")
# fig.savefig("img/violin_go.pdf")
plt.ion()
plt.show()


emg_rows = []

for participant in participants:
    participant_data = {}
    df_part = df.filter(polars.col("Participant") == participant)
    strat = df_part["strategy"].unique()
    strat_dict = {"1": True, "2": True, "3": True, "4": True, "5": True}
    df_block = df_part
    x, y = df_block["IDe"], df_block["MT"]
    try:
        print("yes")
        params, fit = compute_emg_regression_linear_expo_mean(x, y)
        params_gauss, fit_gauss = compute_gaussian_regression_linear_expo_mean(x, y)
        X = sm.add_constant(numpy.asarray(x))
        ols_fit = sm.OLS(numpy.asarray(y), exog=X).fit()
        ols_params = ols_fit.params
        ols_aic = ols_fit.aic
        gauss_aic = 2 * len(fit_gauss.x) + 2 * fit_gauss.fun  # fit.fun returns -ll
        emg_aic = 2 * len(fit.x) + 2 * fit.fun
        R = numpy.exp((emg_aic - ols_aic) / 2)

    except RuntimeError:
        params = [numpy.nan for i in range(5)]

    emg_rows.append(
        [
            participant,
            *params,
            *ols_fit.params,
            numpy.sqrt(ols_fit.scale),
            *params_gauss,
            emg_aic,
            ols_aic,
            gauss_aic,
        ]
    )
    fig, ax = plt.subplots(1, 1)
    ax.plot(
        df_block["IDe"],
        df_block["MT"],
        marker="o",
        linestyle="",
    )
    vec = [
        pearsonr(x, y).statistic,
        spearmanr(x, y).statistic,
        kendalltau(x, y).statistic,
    ]
    ax.set_title(
        rf"P{participant}: r = {vec[0]:.2f}, $\rho$ = {vec[1]:.2f}, $\tau$ = {vec[2]:.2f}",
        fontdict=fontdict,
    )
    vec_x = [numpy.min(numpy.asarray(x)), numpy.max(numpy.asarray(x))]
    ax.plot(vec_x, [params[0] + params[1] * v for v in vec_x], "-", label=f"EMG")
    ax.plot(
        vec_x,
        [params_gauss[0] + params_gauss[1] * v for v in vec_x],
        "-",
        label=f"Gauss QV R = {numpy.exp((emg_aic - gauss_aic) / 2):.2e}",
    )
    ax.plot(
        vec_x,
        [ols_params[0] + ols_params[1] * v for v in vec_x],
        "-",
        label=f"LR  R = {numpy.exp((emg_aic - ols_aic) / 2):.2e}",
    )
    ax.set_xlabel("IDe (bit)")
    ax.set_ylabel("MT (s)")
    ax.legend()
    # plt.ion()
    # plt.show()
    fig.tight_layout()
    if FIG_OUTPUT:
        fig.savefig(
            f"supp_source/gop/pooled_strat_and_r/fitts_gop_{participant}_{r}.pdf"
        )
    plt.close(fig)


emg_pooled_strat_and_r = pandas.DataFrame(
    emg_rows,
    columns=[
        "Participant",
        "emg_beta_0",
        "emg_beta_1",
        "emg_sigma",
        "emg_lambda_0",
        "emg_lambda_1",
        "lr_beta_0",
        "lr_beta_1",
        "lr_sigma",
        "gauss_qv_beta_0",
        "gauss_qv_beta_1",
        "gauss_qv_lambda_0",
        "gauss_qv_lambda_1",
        "emg_aic",
        "lr_aic",
        "gauss_qv_aic",
    ],
)

emg_pooled_strat_and_r["Variance model"] = (
    emg_pooled_strat_and_r["gauss_qv_aic"] - emg_pooled_strat_and_r["lr_aic"]
)
emg_pooled_strat_and_r["Symmetry"] = (
    emg_pooled_strat_and_r["emg_aic"] - emg_pooled_strat_and_r["gauss_qv_aic"]
)
df_emg_aic = emg_pooled_strat_and_r[["Variance model", "Symmetry"]]
df_melted = df_emg_aic.melt()
fig2, ax = plt.subplots(1, 1)
seaborn.violinplot(data=df_melted, x="variable", y="value", ax=ax)
ax.set_xlabel("Modality compared")
ax.set_ylabel("AIC differences")
seaborn.stripplot(
    data=df_melted, x="variable", y="value", color="black", size=5, jitter=True, ax=ax
)
# fig2.savefig("img/violin_go_pooled_r.pdf")
plt.tight_layout()
plt.ion()
plt.show()


df = emg_pooled_strat_and_r[
    ["emg_beta_0", "emg_beta_1", "emg_sigma", "emg_lambda_0", "emg_lambda_1"]
]

_name_map = {
    "emg_beta_0": r"$\beta_0$",
    "emg_beta_1": r"$\beta_1$",
    "emg_sigma": r"$\sigma$",
    "emg_lambda_0": r"$\lambda_0$",
    "emg_lambda_1": r"$\lambda_1$",
}
df = df.rename(columns=_name_map)
ax = seaborn.pairplot(df, kind="reg", diag_kind="kde")
# ax.figure.savefig("img/pairplot_gop.pdf")
plt.ion()
plt.show()
