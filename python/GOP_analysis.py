import emgregs
import polars
import numpy
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import seaborn

plt.style.use("fivethirtyeight")
cm = 1 / 2.54


class UnitData:
    def __init__(self, df):
        self.df = df

    def __str__(self):
        return self.df.__str__()

    def __repr__(self):
        return self.df.__repr__()

    def __len__(self):
        return self.df.__len__()


class GOP_dataset_handler(UnitData):
    def __init__(self, df):
        super().__init__(df)

    def std(self):
        self._std = self.df["Xf"].std()
        return self._std

    def ide(self):
        try:
            self._ide = numpy.log2(1 + 150 / 4.133 / self.std())
        except TypeError:
            self._ide = None
        return self._ide

    def mean_mt(self):
        self._mean_mt = self.df["MT"].mean()
        return self._mean_mt


### Unpacking data

df = polars.read_csv("final_fitts_JG_060323.csv", has_header=True, ignore_errors=True)

participants = df["Participant"].unique().to_list()
strategies = [1, 2, 3, 4, 5]
blocks = [1, 2, 3, 4, 5]


experiment_data = {}

for participant in participants:  # last participant is empty data
    # Separate data per participant
    participant_data = {}
    p = df.filter(polars.col("Participant") == participant)
    for s in strategies:
        s_data = p.filter(polars.col("strategy") == s)
        strategy_data = {}
        for b in blocks:
            b_data = s_data.filter(polars.col("block") == b)
            strategy_data[b] = GOP_dataset_handler(b_data)
        participant_data[s] = strategy_data

    experiment_data[participant] = participant_data


### Plot effective ID plots

fig, axs = plt.subplots(1, 1, figsize=(8 * cm, 16 * cm))

# using plt.rcParams["axes.prop_cycle"].by_key()["color"][:5] and asking chatGPT to complement
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

rows = []

x = []
y = []
hue = []

for participant, value in experiment_data.items():
    for strategy, value in value.items():
        for block, value in value.items():
            if len(value) == 0:
                continue
            try:
                axs.plot(
                    value.ide(), value.mean_mt(), "*", color=colors[int(strategy) - 1]
                )
                x.append((value.ide()))
                y.append(value.mean_mt())
                hue.append(int(strategy))
                # axs[1].plot(
                #     value.ide(),
                #     value.mean_mt(),
                #     "*",
                #     color=colors[int(participant) - 1],
                # )
            except ValueError:
                if value.ide() is None:
                    print("here")
                    pass
                else:  # reraise exception
                    axs.plot(value.ide(), value.mean_mt(), "*")
            rows.append(
                [
                    participant,
                    strategy,
                    block,
                    value.mean_mt(),
                    value.ide(),
                    value.std(),
                ]
            )

axs.set_xlabel(r"$\text{ID}_e (s)$")
axs.set_ylabel("MT (s)")
fig.tight_layout()

fig, ax = plt.subplots(1, 1)
seaborn.scatterplot(x=x, y=y, hue=hue, ax=ax, palette="colorblind")
fig.savefig("img/fitts_ide.pdf")
# plt.ion()
# plt.show()
# exit()

# clear demarcation by strategy, not sure about inter vs intra participant variability. Let's adjust a simple linear model to see effect sizes


fig_all, ax_all = plt.subplots(1, 1, figsize=(8.11, 8.11))
seaborn.scatterplot(
    x=x, y=y, hue=hue, ax=ax_all, zorder=15, palette="colorblind", alpha=1
)
handles, labels = ax_all.get_legend_handles_labels()
new_labels = ["Fast (Emphasis)", "Fast", "Balanced", "Precise", "Precise (Emphasis)"]
ax_all.legend(handles=handles, labels=new_labels, title="Strategy")
ax_all.set_xlabel("IDe (bit)")
ax_all.set_ylabel("mean MT (s)")

# color = colorblind_palette[i]

import statsmodels.formula.api as smf

df = polars.DataFrame(
    rows, schema=["participant", "strategy", "block", "mt", "ide", "std"]
)
lm = smf.mixedlm(formula="mt ~ ide", data=df, groups=df["participant"])
lmfit = lm.fit()
print(lmfit.summary())

fig, ax = plt.subplots(1, 1)
_id = numpy.linspace(2.5, 10, 2)
for i, v in lmfit.random_effects.items():
    y = [lmfit.params[0] + lmfit.params[1] * x + float(v) for x in _id]
    ax.plot(_id, y, "-", color=colors[int(i) - 1])

plt.ion()
plt.show()
# variability of the groups: 0.056, rather small. We may need to look at the effect of outliers.

from scipy.stats import multivariate_normal


def custom_fit(df, strategy, ax, **kwargs):

    fc = kwargs.pop("fc", "None")
    facecolor = kwargs.pop("facecolor", fc)
    ec = kwargs.pop("ec", "r")
    edgecolor = kwargs.pop("edgecolor", ec)
    lw = kwargs.pop("lw", 2)

    df1 = df.filter(polars.col("strategy") == strategy)
    df2 = df1.select(["ide", "mt"])
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
    ax.plot(df1["ide"], df1["mt"], "*")
    return mean, cov, ax, angle, width, height


fig, axs = plt.subplots(1, 5, figsize=(40 * cm, 8 * cm), sharex=True, sharey=True)
# for ax in axs.flat:
#     ax.set_aspect("equal")
# plt.subplots_adjust(left=0.15, right=0.95, top=1, bottom=0.15)
axs[0].set_ylabel("MT (s)")
axs[2].set_xlabel(r"$\text{ID}_e \text{ (bit)}$")
# fig.text(0.5, 0.04, r"$\text{ID}_e \text{ (bit)}$", ha="center", va="center")
# fig.text(0.04, 0.5, "MT (s)", ha="center", va="center", rotation="vertical")


colorblind_palette = seaborn.color_palette("colorblind")

fig2, axs_s = plt.subplots(nrows=1, ncols=5, figsize=(40 * cm, 8 * cm))

mx = []
my = []
vx = []
vy = []
rho = []
for n, strat in enumerate(strategies):
    mean, cov, ax, angle, width, height = custom_fit(df, strat, axs[strat - 1])
    mean, cov, ax, angle, width, height = custom_fit(
        df,
        strat,
        ax_all,
        fc=colorblind_palette[n],
        edgecolor=colorblind_palette[n],
        alpha=0.3,
    )
    # axs_s[0, 0].plot(strat, mean[0], "*")
    # axs_s[0, 1].plot(strat, mean[1], "*")
    # axs_s[1, 0].plot(strat, cov[0, 0], "*")
    # axs_s[1, 1].plot(strat, cov[1, 1], "*")
    # axs_s[1, 2].plot(strat, cov[1, 0] / numpy.sqrt(cov[0, 0] * cov[1, 1]), "*")

    mx.append(mean[0])
    my.append(mean[1])
    vx.append(numpy.sqrt(cov[0, 0]))
    vy.append(numpy.sqrt(cov[1, 1]))
    rho.append(cov[1, 0] / numpy.sqrt(cov[0, 0] * cov[1, 1]))

fig_all.tight_layout()
fig_all.savefig("img/fitts_ide_go_with_ellipse.pdf")
import seaborn
import statsmodels.api as sm

strat = [1, 2, 3, 4, 5]
strat = strat - numpy.mean(strat)
strat = 2 * strat / (numpy.max(strat) - numpy.min(strat))

fontdict = {
    "fontsize": 14,
}
seaborn.regplot(x=strat, y=mx, ax=axs_s[0])
_x = sm.add_constant(strat)
model = sm.OLS(mx, _x).fit()
print(model.summary().as_latex())
axs_s[0].set_title(
    f"{model.params[0]:.2f} + {model.params[1]:.2f}strat, r² = {model.rsquared:.2f}",
    fontdict=fontdict,
)
axs_s[0].set_ylabel(r"$\mu_i$")

seaborn.regplot(x=strat, y=my, ax=axs_s[1])
model = sm.OLS(my, _x).fit()
print(model.summary().as_latex())

axs_s[1].set_title(
    f"{model.params[0]:.2f} + {model.params[1]:.2f}strat, r² = {model.rsquared:.2f}",
    fontdict=fontdict,
)

axs_s[1].set_ylabel(r"$\mu_t$")


seaborn.regplot(x=strat, y=vx, ax=axs_s[2])
model = sm.OLS(vx, _x).fit()
print(model.summary().as_latex())

axs_s[2].set_title(
    f"{model.params[0]:.2f} + {model.params[1]:.2f}strat, r² = {model.rsquared:.2f}",
    fontdict=fontdict,
)
axs_s[2].set_ylabel(r"$\sigma_i$")
axs_s[2].set_xlabel("Strategy (numerical)")

seaborn.regplot(x=strat, y=vy, ax=axs_s[3])
model = sm.OLS(vy, _x).fit()
print(model.summary().as_latex())

axs_s[3].set_title(
    f"{model.params[0]:.2f} + {model.params[1]:.2f}strat, r² = {model.rsquared:.2f}",
    fontdict=fontdict,
)
axs_s[3].set_ylabel(r"$\sigma_t$")


seaborn.regplot(x=strat, y=rho, ax=axs_s[4])
model = sm.OLS(rho, _x).fit()
print(model.summary().as_latex())

axs_s[4].set_title(
    f"{model.params[0]:.2f} + {model.params[1]:.2f}strat, r² = {model.rsquared:.2f}",
    fontdict=fontdict,
)
axs_s[4].set_ylabel(r"$\rho$")

fig.tight_layout()
fig.savefig("img/mean_cov.pdf")
fig2.tight_layout()
fig2.savefig("img/mean_cov_strat.pdf")
plt.show()


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
