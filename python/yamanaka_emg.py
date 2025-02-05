import polars, pandas
import numpy
import matplotlib.pyplot as plt
import seaborn
from emg_arbitrary_variance import (
    compute_emg_regression_linear_expo_mean,
    compute_gaussian_regression_linear_expo_mean,
)
import statsmodels.api as sm

plt.style.use(["fivethirtyeight"])

path = "/home/juliengori/Documents/VC/gen_pointing_stat/python/ToJulien_CHI2023Yamanaka/Exp1BubbleCursor/allData.csv"

df = polars.read_csv(path)
pointing_data = df.filter(polars.col("isBubble") == False)
pointing_data = pointing_data.with_columns(
    (numpy.log2(1 + polars.col("A") / polars.col("W"))).alias("ID")
)
participants = pointing_data["participantNumber"].unique()
strategies = pointing_data["bias"].unique()
Ds = pointing_data["A"].unique()
Ws = pointing_data["W"].unique()
WeW = pointing_data["EW/W"].unique()

t0 = pointing_data[0]["targetX"]

rows = []
k = 0
for _d in Ds:
    for _w in Ws:
        for p in participants:
            for s in strategies:
                for nw, _wew in enumerate(WeW):
                    p_data = pointing_data.filter(
                        (polars.col("participantNumber") == p)
                        & (polars.col("A") == _d)
                        & (polars.col("W") == _w)
                        & (polars.col("bias") == s)
                        & (polars.col("EW/W") == _wew)
                    )

                    # we = (
                    #     4.133
                    #     * numpy.sqrt(
                    #         (p_data["x"] - p_data["targetX"]) ** 2
                    #         + (p_data["y"] - p_data["targetY"]) ** 2
                    #     )
                    #     .sqrt()
                    #     .sum()
                    #     / (len(p_data))
                    # )
                    A = p_data["A"].unique().item()
                    W = p_data["W"].unique().item()
                    # IDe = numpy.log2(1 + A / we)
                    ID = p_data["ID"].unique().item()

                    initial_position = (0, 0)

                    X = []
                    Y = []

                    for n, row in enumerate(p_data.iter_rows()):
                        x0, y0 = initial_position
                        tx = row[12]
                        ty = row[13]
                        x = row[8]
                        y = row[9]
                        alpha = numpy.arctan2(ty - y0, tx - x0)
                        x, y = x - x0, y - y0
                        R = numpy.array(
                            [
                                [numpy.cos(alpha), numpy.sin(alpha)],
                                [-numpy.sin(alpha), numpy.cos(alpha)],
                            ]
                        )
                        v = numpy.array([x, y])
                        x, y = R @ v
                        # if (k==123) and (nw ==2):
                        #     print(nw, n,x,y)
                        X.append(x)
                        Y.append(y)
                        initial_position = (row[12], row[13])
                    # if (k==123) and (nw == 0):
                    #     exit()

                    X = numpy.array(X[1:])
                    stdX = (X - numpy.mean(X)) / numpy.std(X)
                    keep = numpy.where(numpy.abs(stdX) < 2.5)[0]
                    sigma_x = numpy.std(X[keep])
                    we = 4.133 * sigma_x
                    IDe = numpy.log2(1 + A / we)
                    for n, row in enumerate(p_data.iter_rows()):
                        if n in keep + 1:
                            # continue
                            rows.append([p, k, s, ID, IDe, row[11], A, W])

                k += 1


df = polars.DataFrame(
    rows, schema=["participant", "set", "strategy", "ID", "IDe", "MT", "A", "W"]
)


df = df.with_columns(
    polars.col("ID").round(3)
)  # floating point precision meant 9 levels for ID instead of 6

emg_rows = []

for (p, s), group in df.group_by(["participant", "strategy"]):
    x, y = group["IDe"], group["MT"]
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
            p,
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

emg_pooled_DW = pandas.DataFrame(
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

emg_pooled_DW["Variance model"] = (
    emg_pooled_DW["gauss_qv_aic"] - emg_pooled_DW["lr_aic"]
)
emg_pooled_DW["Symmetry"] = emg_pooled_DW["emg_aic"] - emg_pooled_DW["gauss_qv_aic"]
df_emg_aic = emg_pooled_DW[["Variance model", "Symmetry"]]
df_melted = df_emg_aic.melt()
fig, ax = plt.subplots(1, 1)
seaborn.violinplot(data=df_melted, x="variable", y="value", ax=ax)
seaborn.stripplot(
    data=df_melted, x="variable", y="value", color="black", size=5, jitter=True, ax=ax
)
ax.set_xlabel("Modality compared")
ax.set_ylabel("AIC differences")
# fig.savefig("img/violin_yamanaka_pooled_dw.pdf")
plt.tight_layout()
plt.ion()
plt.show()

fig2, axs = plt.subplots(1, 5)
for ni, i in enumerate(
    ["emg_beta_0", "emg_beta_1", "emg_sigma", "emg_lambda_0", "emg_lambda_1"]
):
    seaborn.scatterplot(data=emg_pooled_DW, x="Strategy", y=i, ax=axs[ni])
plt.ion()
plt.show()
