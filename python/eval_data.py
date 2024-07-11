import numpy
import seaborn
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import scipy.stats as stats

cm = 1 / 2.54


def gaussian_fit(x, y, ax=None):
    data = numpy.array(object=[numpy.asarray(x), numpy.asarray(y)]).T
    mean, cov = stats.multivariate_normal.fit(data)
    eigenvalues, eigenvectors = numpy.linalg.eigh(cov)
    angle = numpy.degrees(numpy.arctan2(*eigenvectors[:, 0][::-1]))
    width, height = 2 * numpy.sqrt(5.991 * eigenvalues)
    if ax is not None:
        ellipse = Ellipse(
            xy=mean,
            width=width,
            height=height,
            angle=angle,
            edgecolor="r",
            fc="None",
            lw=2,
        )
        ax.add_patch(ellipse)
        ax.plot(x, y, "*")
    return mean, cov, ax, angle, width, height


def evaluate_data_mean_mvgauss(
    all_data_dict, strategies=[-1, -0.5, 0, 0.5, 1], axs=None
):

    if axs is None:
        fig1, axs = plt.subplots(
            nrows=1, ncols=5, sharex=True, sharey=True, figsize=(40 * cm, 8 * cm)
        )
        fig2, axs2 = plt.subplots(
            nrows=1, ncols=5, sharex=True, figsize=(40 * cm, 8 * cm)
        )

    mx = []
    my = []
    vx = []
    vy = []
    rho = []
    for n, strat in enumerate(strategies):
        x, y = all_data_dict[n]
        meanx = [numpy.mean(_x) for _x in x]
        meany = [numpy.mean(_y) for _y in y]

        mean, cov, ax, angle, width, height = gaussian_fit(meanx, meany, axs[n])

        mx.append(mean[0])
        my.append(mean[1])
        vx.append(numpy.sqrt(cov[0, 0]))
        vy.append(numpy.sqrt(cov[1, 1]))
        rho.append(cov[1, 0] / numpy.sqrt(cov[0, 0] * cov[1, 1]))

    fontdict = {
        "fontsize": 14,
    }

    for n, (elem, ylabel) in enumerate(
        zip(
            [mx, my, vx, vy, rho],
            [r"$\mu_i$", r"$\mu_t$", r"$\sigma_i$", r"$\sigma_t$", r"$\rho$"],
        )
    ):

        seaborn.regplot(x=strategies, y=elem, ax=axs2[n])
        _x = sm.add_constant(strategies)
        model = sm.OLS(elem, _x).fit()
        axs2[n].set_title(
            f"{model.params[0]:.2f} + {model.params[1]:.2f}strat, r² = {model.rsquared:.2f}",
            fontdict=fontdict,
        )
        axs2[n].set_ylabel(ylabel)

    axs2[2].set_xlabel("Strategy (numerical)")

    return (fig1, axs), (fig2, axs2)
