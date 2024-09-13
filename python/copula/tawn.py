import numpy
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy


def clayton_copula_pdf_R(u1, u2, alpha):

    return (
        (1 + (u1 ** (-alpha) - 1 + u2 ** (-alpha) - 1)) ** (((-1 / alpha) - 1) - 1)
        * (((-1 / alpha) - 1) * (u2 ** ((-alpha) - 1) * (-alpha)))
        * ((-1 / alpha) * (u1 ** ((-alpha) - 1) * (-alpha)))
    )


def density_bivariate(
    X,
    copula=dict(name="clayton", params=(2,)),
    marginals=[dict(name="norm", loc=0, scale=1), dict(name="norm", loc=0, scale=1)],
):
    # X is an array where shape ends with 2

    margs = []
    for marg in marginals:
        name = marg.pop("name")
        distrib = getattr(scipy.stats, name)
        margs.append(distrib(**marg))

    U = numpy.zeros(X.shape)
    U[..., 0] = margs[0].cdf(X[..., 0])
    U[..., 1] = margs[1].cdf(X[..., 1])

    density_marginals = margs[0].pdf(X[..., 0]) * margs[1].pdf(X[..., 1])
    cop_name = copula.pop("name")
    if cop_name == "clayton":
        cop = clayton_copula_pdf_R(U[..., 0], U[..., 1], *copula["params"])
    elif cop_name == "tawn_theta2__81_delta0__81":
        cop = rotated_tawn_typeI_theta2__81_delta0__81_pdf(U)
    else:
        raise NotImplementedError
    return cop * density_marginals


# load tawn_pdf data
import pandas

df = pandas.read_csv("tawn_pdf_theta2__81_delta0__81.csv")

u = numpy.linspace(0.01, 0.99, 500)
v = numpy.linspace(0.01, 0.99, 500)
U, V = numpy.meshgrid(u, v)
U = U.ravel()
V = V.ravel()
lin_grid = numpy.concatenate(
    (U.reshape(-1, 1), V.reshape(-1, 1), numpy.array(df["x"]).reshape(-1, 1)), axis=1
)
rotated_tawn_typeI_theta2__81_delta0__81_pdf = scipy.interpolate.LinearNDInterpolator(
    lin_grid[:, :2], lin_grid[:, 2]
)

# Define the Clayton copula parameters
theta = 2  # copula parameter, theta > 0

# Define the grid for the contour plot
x = numpy.linspace(-3, 3, 100)  # Avoid 0 and 1 to prevent division by zero
y = numpy.linspace(-3, 3, 100)

X = numpy.array([x, y])


X, Y = numpy.meshgrid(x, y)

pos = numpy.dstack((X, Y))

# copula = dict(name="clayton", params=(2,))
# marginals = [dict(name="norm", loc=0, scale=1), dict(name="norm", loc=0, scale=1)]
# dcopula_bis = density_bivariate(pos, copula=copula, marginals=marginals)

copula = dict(name="tawn_theta2__81_delta0__81")
marginals = [dict(name="norm", loc=0, scale=1), dict(name="norm", loc=0, scale=1)]
tawn_dcopula_ = density_bivariate(pos, copula=copula, marginals=marginals)

# Plot the contour plot
plt.figure(figsize=(8, 6))
# contour = plt.contour(X, Y, Z, levels=numpy.linspace(0.02, 0.2, 9), cmap="viridis")
contour = plt.contour(X, Y, tawn_dcopula_, levels=20, cmap="viridis")
plt.colorbar(contour)
plt.title(
    "Contour Plot of a bivariate with Clayton Copula and standard Normal marginals"
)
plt.xlabel("U1")
plt.ylabel("U2")
plt.grid(True)
plt.show()
