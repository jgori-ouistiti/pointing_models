import polars, pandas
import numpy
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import seaborn
import statsmodels.api as sm
from scipy.stats import multivariate_normal, pearsonr, kendalltau, spearmanr
import pickle
from emg_arbitrary_variance import compute_emg_regression_linear_expo_mean


plt.style.use("fivethirtyeight")


### Controls
FIG_OUTPUT = False

### Unpacking data


# df = polars.read_csv("fitts_csv_GOP.csv", has_header=True, ignore_errors=True)
# df = df.filter(polars.col("Participant") != 9)

df = pandas.read_csv("fitts_csv_GOP.csv")
df = df[df["Participant"] != 9]


def remove_outliers(group, column):
    mean = group[column].mean()
    std = group[column].std()
    return group[numpy.abs(group[column] - mean) <= 2.5 * std]


filtered_df = (
    df.groupby(["IDe"]).apply(remove_outliers, column="MT").reset_index(drop=True)
)
# 5937 - 5858 = 79 points removed

fig, axs = plt.subplots(2, 5, sharex=True, sharey=True)
x_min = 100
x_max = -100
y_min = 100
y_max = 100
for i in range(1, 6):
    df_strat = df[df["strategy"] == i]
    df_filtered_strat = filtered_df[filtered_df["strategy"] == i]
    ax = seaborn.scatterplot(df_strat, x="IDe", y="MT", ax=axs[0, i - 1])
    ax = seaborn.scatterplot(df_filtered_strat, x="IDe", y="MT", ax=axs[1, i - 1])
    # x_min = min(x_min, ax.get_xlim()[0])
    # x_max = min(x_max, ax.get_xlim()[1])
    # y_min = min(y_min, ax.get_ylim()[0])
    # y_max = min(y_max, ax.get_ylim()[1])

# for ax in axs:
#     ax.set_xlim([x_min, x_max])
#     ax.set_
plt.ion()
plt.show()

df = polars.DataFrame(filtered_df)

participants = df["Participant"].unique()
strategies = df["strategy"].unique()
repetitions = df["repetition"].unique()

k = 0
for np, p in enumerate(participants):
    df_part = df.filter(polars.col("Participant") == p)
    df_part.write_csv(f"exchange/gop_part/{np}.csv")
    for ns, s in enumerate(strategies):
        df_strat = df_part.filter(polars.col("strategy") == s)
        df_strat.write_csv(f"exchange/gop_part_strat/{np}_{ns}.csv")
        for nr, r in enumerate(repetitions):
            df_repet = df_strat.filter(polars.col("repetition") == r)
            df_repet.write_csv(f"exchange/gop_part_strat_repet/{np}_{ns}_{nr}.csv")
            k += 1


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

experiment_data = {}

association_rows = []
emg_rows = []

for participant in participants:
    participant_data = {}
    df_part = df.filter(polars.col("Participant") == participant)
    fig, ax = plt.subplots(1, 1)
    strat = df_part["strategy"].unique()
    strat_dict = {"1": True, "2": True, "3": True, "4": True, "5": True}
    vec = [
        pearsonr(df_part["IDe"], df_part["MT"]).statistic,
        spearmanr(df_part["IDe"], df_part["MT"]).statistic,
        kendalltau(df_part["IDe"], df_part["MT"]).statistic,
    ]

    association_rows.append(vec)
    for s in strat:
        strategy_data = {}
        df_strat = df_part.filter(polars.col("strategy") == s)
        repetitions = df_part["repetition"].unique()
        for r in repetitions:
            df_block = df_strat.filter(polars.col("repetition") == r)
            strategy_data["r"] = df_block

            strat = int(s)

            x, y = df_block["IDe"], df_block["MT"]
            try:
                params, fit = compute_emg_regression_linear_expo_mean(x, y)

            except RuntimeError:
                params = [numpy.nan for i in range(5)]

            emg_rows.append([participant, s, r, *params])

            ax.plot(
                df_block["IDe"],
                df_block["MT"],
                color=colors[strat],
                marker="o",
                linestyle="",
            )
            if strat_dict[str(s)]:
                ax.lines[-1].set_label(label_mapping[str(s)])
                strat_dict[str(s)] = False
            ax.plot(
                numpy.mean(numpy.asarray(df_block["IDe"])),
                numpy.mean(numpy.asarray(df_block["MT"])),
                marker="D",
                color="k",
                ms=8,
            )
        participant_data[s] = strategy_data
    experiment_data[participant] = participant_data

    ax.set_xlabel("IDe (bit)")
    ax.set_ylabel("MT (s)")
    ax.legend()
    fig.tight_layout()
    if FIG_OUTPUT:
        fig.savefig(f"supp_source/gop/fitts_gop_{participant}.pdf")
    plt.close(fig)

emg = polars.DataFrame(
    emg_rows,
    schema=[
        "Participant",
        "Strategy",
        "Repetition",
        "beta_0",
        "beta_1",
        "sigma",
        "lambda_0",
        "lambda_1",
    ],
)

with open("gop_emg_params_all_p_s_r.pkl", "wb") as _file:
    pickle.dump(emg, _file)


association_rows = []
fig, axs = plt.subplots(1, 5)
for participant in participants:
    participant_data = {}
    df_part = df.filter(polars.col("Participant") == participant)

    repetitions = df_part["repetition"].unique()
    for nr, r in enumerate(repetitions):
        df_block = df_part.filter(polars.col("repetition") == r)
        vec = [
            participant,
            r,
            pearsonr(df_block["IDe"], df_block["MT"]).statistic,
            spearmanr(df_block["IDe"], df_block["MT"]).statistic,
            kendalltau(df_block["IDe"], df_block["MT"]).statistic,
        ]

        association_rows.append(vec)

        axs[nr].plot(
            df_block["IDe"],
            df_block["MT"],
            marker="o",
            linestyle="",
        )


association = polars.DataFrame(
    association_rows,
    schema=["Participant", "repetition", "r", "rho", "tau"],
)


plt.show()
with open("gop_association_agg_strategy.pkl", "wb") as _file:
    pickle.dump(association, _file)

association_rows = []
fig, axs = plt.subplots(1, 5)
for participant in participants:
    participant_data = {}
    df_part = df.filter(polars.col("Participant") == participant)

    strategies = df_part["strategy"].unique()
    for ns, s in enumerate(strategies):
        df_block = df_part.filter(polars.col("strategy") == s)
        vec = [
            participant,
            s,
            pearsonr(df_block["IDe"], df_block["MT"]).statistic,
            spearmanr(df_block["IDe"], df_block["MT"]).statistic,
            kendalltau(df_block["IDe"], df_block["MT"]).statistic,
        ]

        association_rows.append(vec)
        axs[ns].plot(
            df_block["IDe"],
            df_block["MT"],
            marker="o",
            linestyle="",
        )


association = polars.DataFrame(
    association_rows,
    schema=["Participant", "strategy", "r", "rho", "tau"],
)
plt.show()

fig, axs = plt.subplots(1, 3)
seaborn.histplot(association, x="r", hue="strategy", ax=axs[0], label="Pearson r")
seaborn.histplot(
    association, x="rho", hue="strategy", ax=axs[1], label=r"Spearman $\rho$"
)
seaborn.histplot(
    association, x="tau", hue="strategy", ax=axs[2], label=r"Kendall $\tau$"
)
plt.tight_layout()
plt.ion()
# plt.savefig("img/association_gop.pdf")
with open("gop_association_agg_strategy.pkl", "wb") as _file:
    pickle.dump(association, _file)
### Plot effective ID plots
cm = 2.54


def custom_fit(df, strategy, ax, **kwargs):

    fc = kwargs.pop("fc", "None")
    facecolor = kwargs.pop("facecolor", fc)
    ec = kwargs.pop("ec", "r")
    edgecolor = kwargs.pop("edgecolor", ec)
    lw = kwargs.pop("lw", 2)

    df1 = df.filter(polars.col("strategy") == strategy)
    df2 = df1.select(["IDe", "MT"])
    mean, cov = multivariate_normal.fit(df2)
    eigenvalues, eigenvectors = numpy.linalg.eigh(cov)
    angle = numpy.degrees(numpy.arctan2(*eigenvectors[:, 0][::-1]))
    width, height = 2 * numpy.sqrt(5.991 * eigenvalues)

    ellipse = Ellipse(
        xy=mean,
        width=width,
        height=height,
        angle=angle,
        edgecolor=edgecolor,
        facecolor=facecolor,
        lw=lw,
        **kwargs,
    )
    ax.add_patch(ellipse)
    ax.plot(df1["IDe"], df1["MT"], "*")
    return mean, cov, ax, angle, width, height


seaborn.scatterplot(df, x="IDe", y="MT", hue="strategy", palette="colorblind")
plt.tight_layout()


if FIG_OUTPUT:
    fig.savefig("img/fitts_ide.pdf")
plt.show()
plt.close(fig)


blocks = df["IDe"].unique()
strategies = df["strategy"].unique()


rows_all = []
for b in blocks:
    df_block = df.filter(polars.col("IDe") == b)
    df_block["strategy"].unique()[0]
    rows_all.append(
        [
            df_block["strategy"].unique()[0],
            numpy.mean(numpy.asarray(df_block["IDe"])),
            numpy.mean(numpy.asarray(df_block["MT"])),
        ]
    )


df_mean = polars.DataFrame(rows_all, schema=["strategy", "IDe", "MT"])

colorblind_palette = seaborn.color_palette(palette="colorblind")

fig, ax = plt.subplots(1, 1)
figs, axs = plt.subplots(1, 5)

# seaborn.scatterplot(
#     df_mean, x="IDe", y="MT", hue="strategy", palette="colorblind", ax=ax
# )


mx = []
my = []
vx = []
vy = []
rho = []
for n, strat in enumerate(strategies):
    mean, cov, _, angle, width, height = custom_fit(df_mean, strat, axs[strat - 1])
    mean, cov, ax, angle, width, height = custom_fit(
        df_mean,
        strat,
        ax,
        fc=colorblind_palette[n],
        edgecolor=colorblind_palette[n],
        alpha=0.3,
    )
    ax.lines[-1].set_label(label_mapping[str(strat)])
    mx.append(mean[0])
    my.append(mean[1])
    vx.append(numpy.sqrt(cov[0, 0]))
    vy.append(numpy.sqrt(cov[1, 1]))
    rho.append(cov[1, 0] / numpy.sqrt(cov[0, 0] * cov[1, 1]))

ax.set_xlabel(r"$\text{ID}_e$" + " (bit)")
ax.set_ylabel(r"$\overline{\text{MT}}$ (s)")
ax.legend()
plt.ion()
fig.tight_layout()
if FIG_OUTPUT:
    fig.savefig("img/fitts_ide_go_with_ellipse.pdf")
plt.show()

## here

df = pandas.DataFrame(
    {
        "mu_i": mx,
        "mu_t": my,
        "sigma_i": vx,
        "sigma_t": vy,
        "rho": rho,
        "strategy": [-1, -0.5, 0, 0.5, 1],
    }
)

fig2, axs_s = plt.subplots(1, 5)


## Naive version
import seaborn
import statsmodels.api as sm

strat = [1, 2, 3, 4, 5]
strat = strat - numpy.mean(strat)
strat = 2 * strat / (numpy.max(strat) - numpy.min(strat))

fontdict = {
    "fontsize": 14,
}
seaborn.scatterplot(x=strat, y=mx, ax=axs_s[0], s=50)
import statsmodels.formula.api as smf

model = smf.ols(formula="mu_i~1", data=df)  # AIC = 21
model = smf.ols(formula="mu_i~strategy", data=df)  # AIC = 0.62
model = model.fit()
print(model.summary())
axs_s[0].plot(strat, [model.params[0] + model.params[1] * s for s in strat], "-")
axs_s[0].set_title(
    f"{model.params[0]:.2f} + {model.params[1]:.2f}strat, r² = {model.rsquared:.2f}",
    fontdict=fontdict,
)
axs_s[0].set_ylabel(r"$\mu_i$")

seaborn.scatterplot(x=strat, y=my, ax=axs_s[1], s=50)

model = smf.ols(formula="mu_t~1", data=df)  # AIC = 8.5
model = smf.ols(formula="mu_t~strategy", data=df)  # AIC = -4
model = model.fit()

print(model.summary())
axs_s[1].plot(strat, [model.params[0] + model.params[1] * s for s in strat], "-")

axs_s[1].set_title(
    f"{model.params[0]:.2f} + {model.params[1]:.2f}strat, r² = {model.rsquared:.2f}",
    fontdict=fontdict,
)

axs_s[1].set_ylabel(r"$\mu_t$")


seaborn.scatterplot(x=strat, y=vx, ax=axs_s[2], s=50)
model = smf.ols(formula="sigma_i~strategy", data=df)  # AIC = 1
model = smf.ols(formula="sigma_i~1", data=df)  # AIC = 3
model = model.fit()
exit()

print(model.summary())
axs_s[2].plot(strat, [model.params[0] for s in strat], "-")

axs_s[2].set_title(
    f"{model.params[0]:.2f}",
    fontdict=fontdict,
)
axs_s[2].set_ylabel(r"$\sigma_i$")
axs_s[2].set_xlabel("Strategy (numerical)")

seaborn.scatterplot(x=strat, y=vy, ax=axs_s[3], s=50)
model = smf.ols(formula="sigma_t~1", data=df)  # AIC = -10
model = smf.ols(formula="sigma_t~strategy", data=df)  # AIC = -17
model = model.fit()
print(model.summary())
axs_s[3].plot(strat, [model.params[0] + model.params[1] * s for s in strat], "-")

axs_s[3].set_title(
    f"{model.params[0]:.2f} + {model.params[1]:.2f}strat, r² = {model.rsquared:.2f}",
    fontdict=fontdict,
)
axs_s[3].set_ylabel(r"$\sigma_t$")


seaborn.scatterplot(x=strat, y=rho, ax=axs_s[4], s=50)
model = smf.ols(formula="rho~strategy", data=df)  # AIC = -3
model = smf.ols(formula="rho~1", data=df)  # AIC = -1
model = model.fit()
print(model.summary())
axs_s[4].plot(strat, [model.params[0] for s in strat], "-")

axs_s[4].set_title(
    f"{model.params[0]:.2f}",
    fontdict=fontdict,
)
axs_s[4].set_ylabel("r")
plt.ion()
plt.show()
plt.tight_layout()
exit()
fig.tight_layout()
if FIG_OUTPUT:
    fig.savefig("img/mean_cov.pdf")
fig2.tight_layout()
if FIG_OUTPUT:
    fig2.savefig("img/mean_cov_strat.pdf")


# This works quite well, and we can see the ellipse evolve over strategies. We now have as coarse model:


# WHO model

fig, axs = plt.subplots(1, 1)
axs.scatter(
    df["mt"],
    df["std"] / 150,
    c=[colors[i] for i in df["strategy"]],
)

# try inverse it for better inference

fig, axs = plt.subplots(1, 1)
y = numpy.array(df["std"])
x = numpy.array(df["mt"])
w = 1 / y

axs.scatter(
    x,
    w,
    c=[colors[i] for i in df["strategy"]],
)


#### ================== Below, I started a transformation on the WHo model, but let's skip it for now. The idea is to plot 1/y as function of x, which gives a nice linear model with exponential noise with quadratic standard deviation (as function of x, and I could fit directly the EMG regression to it, but then I probably could not connect the data to the WHo model, so it sort of defeats the purpose in the first place.

# # it looks like an expoential noise. To see if it could work, let's check the mean to variance relationship
# bin_edges = numpy.quantile(x, numpy.linspace(0, 1, 11))
# bin_indices = numpy.digitize(x, bin_edges)
# bin_x_means = []
# bin_w_means = []
# bin_w_stds = []
# for i in range(1, 11):
#     bin_mask = bin_indices == i
#     bin_x_means.append(numpy.mean(x[bin_mask]))
#     bin_w_means.append(numpy.mean(w[bin_mask]))
#     bin_w_stds.append(numpy.std(w[bin_mask]))

# fig, axs = plt.subplots(1, 2)
# axs[0].plot(bin_x_means, bin_w_means, "*")
# axs[0].plot(bin_x_means, bin_w_stds, "*")
# axs[1].scatter(bin_w_means, bin_w_stds)
# axs[1].plot([0, 5], [0, 5], "r-")

# # it works quite well,


### ==== below some more WHo stuff, but it's a little cumbersome to use ====
# import scipy.optimize as opti


# def hard_hull(_mt, _cv):
#     optix = []
#     optiy = []
#     WHo_data = sorted(zip(_mt, _cv), key=lambda t: t[0])
#     xbest = 0
#     ybest = 1e6
#     for x, y in WHo_data:
#         if y < ybest:
#             optix.append(x)
#             optiy.append(y)
#             ybest = y
#     return optix, optiy


# def WHo_fit(x, y, theta0, _niter=20):
#     def WHo_cost(Theta, x, y):
#         x0, y0, alpha, k = Theta
#         if x0 < 0 or y0 < 0 or alpha < 0 or alpha > 1 or k < 0:
#             return 1e6
#         out = 0
#         for i, v in enumerate(x):
#             t_x = (k * (y[i] - y0) ** (alpha - 1)) ** (1 / alpha) + x0
#             c_x = (v - t_x) ** 2
#             t_y = (k * (v - x0) ** (-alpha)) ** (1 / (1 - alpha)) + y0
#             c_y = (y[i] - t_y) ** 2
#             out += c_y + c_x
#         return out

#     bounds = [(0, 3), (0, 1), (0, 1), (0, 1)]
#     res = opti.basinhopping(
#         func=WHo_cost,
#         x0=theta0,
#         niter=_niter,
#         minimizer_kwargs={
#             # "method": "L-BFGS-B",
#             "method": "Nelder-Mead",
#             "args": (x, y),
#             "bounds": bounds,
#             "options": {"maxiter": 1000, "disp": 1},
#         },
#     )
#     return res


# def WHo_fig(
#     theta0,
#     MAX_WINDOW_X,
#     MAX_WINDOW_Y,
#     krange=[0.1, 0.2, 0.3, 0.4],
#     betarange=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
#     print_betas=1,
#     print_ks=1,
#     ax=None,
# ):
#     sec_y_ticks = []
#     sec_y_ticks_labels = []
#     sec_x_ticks = []
#     sec_x_ticks_labels = []
#     ter_x_ticks = []
#     ter_x_ticks_labels = []

#     x0, y0, alpha, k = theta0

#     _abs = numpy.linspace(x0, MAX_WINDOW_X, 400)
#     y = [(k * (v - x0) ** (-alpha)) ** (1 / (1 - alpha)) + y0 for v in _abs]
#     if ax is None:
#         fig = plt.figure()
#         ax = fig.add_subplot(111)
#     ax.grid(which="both", axis="both", linewidth=0.5, linestyle="--", alpha=0.4)
#     ax.plot(_abs, y, "r-", lw=2)
#     # ax.plot(_abs,y, 'b--', lw = 2)
#     if print_ks:
#         for nk, _k in enumerate(krange):
#             y = [(_k * (v - x0) ** (-alpha)) ** (1 / (1 - alpha)) + y0 for v in _abs]
#             ax.plot(_abs, y, "k-", lw=1, alpha=0.8, color="#ad4718")
#             sec_x_ticks.append(
#                 (MAX_WINDOW_Y - y0) ** ((alpha - 1) / alpha) * _k ** (1 / alpha) + x0
#             )
#             sec_x_ticks_labels.append(str(_k))
#     if print_betas:
#         for nk, _beta in enumerate(betarange):
#             y = [
#                 (1 - alpha) / alpha * (v - x0) / (1 - _beta) * _beta + y0 for v in _abs
#             ]
#             ax.plot(_abs, y, "-", lw=1, alpha=0.8, color="#022c70")
#             if max(y) < MAX_WINDOW_Y:
#                 max_tmp = _abs[-1]
#                 sec_y_ticks.append(
#                     (1 - alpha) / alpha * (max_tmp - x0) / (1 - _beta) * _beta + y0
#                 )
#                 sec_y_ticks_labels.append(str(_beta))
#             else:
#                 pass
#                 ter_x_ticks.append(
#                     (MAX_WINDOW_Y - y0) * (1 - _beta) / _beta / (1 - alpha) * alpha + x0
#                 )
#                 ter_x_ticks_labels.append(str(_beta))

#     ax.set_ylim([-0.15 * MAX_WINDOW_Y, MAX_WINDOW_Y])
#     ax.text(
#         0,
#         -0.10 * MAX_WINDOW_Y,
#         r"$x_0 = {{{:.3f}}}, \quad y_0 = {{{:.3f}}}, \quad \alpha = {{{:.3f}}}, \quad k = {{{:.3f}}}$".format(
#             *theta0
#         ),
#         va="baseline",
#         ha="left",
#     )
#     ax.set_xlim([-0.05 * MAX_WINDOW_X, max(_abs)])
#     if print_betas:
#         ax.text(
#             MAX_WINDOW_X * 0.95,
#             MAX_WINDOW_Y * 0.95,
#             r"$iso-\beta$",
#             color="#022c70",
#             ha="right",
#             va="top",
#         )
#     if print_ks:
#         ax.text(
#             x0,
#             MAX_WINDOW_Y * 0.95,
#             r"$iso-k$",
#             rotation="vertical",
#             va="top",
#             ha="right",
#             color="#ad4718",
#         )
#     ax.set_xlabel(r"$\mu_T$")
#     ax.set_ylabel(r"$\sigma_A/\mu_A$")
#     ax.get_ylim()

#     ax_bk = ax.twinx()
#     ax_bk.grid(False)
#     ax_bk.set_ylim(ax.get_ylim())
#     ax_bk.set_yticks(sec_y_ticks)
#     ax_bk.set_yticklabels(sec_y_ticks_labels, fontsize=10, color="#022c70")
#     ax_bkbis = ax.twiny()
#     ax_bkbis.grid(False)
#     ax_bkbis.set_xlim(ax.get_xlim())
#     ax_bkbis.set_xticks(sec_x_ticks)
#     ax_bkbis.set_xticklabels(sec_x_ticks_labels, fontsize=10, color="#ad4718")

#     ax_bkter = ax.twiny()
#     ax_bkter.grid(False)
#     ax_bkter.set_xlim(ax.get_xlim())
#     ax_bkter.set_xticks(ter_x_ticks)
#     ax_bkter.set_xticklabels(ter_x_ticks_labels, fontsize=10, color="#022c70")

#     return ax


# fig, ax = plt.subplots(1, 1)
# w = y / 150
# optix, optiw = hard_hull(x, w)
# ax.scatter(x, w, s=1)
# ax.scatter(optix, optiw, s=5, c="r")

# theta0 = [numpy.min(x), numpy.min(w) / 2, 0.8, 0.2]
# res = WHo_fit(optix, optiw, theta0)
# # res = WHo_fit(x, w, theta0)

# theta0 = list(res.x)
# # theta0 = [0, 0, 0.8, 0.2]

# WHo_fig(theta0, 3, MAX_WINDOW_Y=0.2, ax=ax)

# plt.ion()
# plt.show()
