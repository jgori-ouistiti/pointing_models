import pickle
import seaborn
import pandas
import matplotlib.pyplot as plt

plt.style.use(["fivethirtyeight"])

with open("jgp_parameters.pkl", "rb") as _file:
    parameters = pickle.load(_file)

# shape = (Np X Ni x [beta0, beta1, sigma, lambda0, lambda1])
dimp, dimi, dimtheta = parameters.shape
df = pandas.DataFrame(
    parameters.transpose(0, 1, 2).reshape(dimp * dimi, dimtheta),
    columns=[r"$\beta_0$", r"$\beta_1$", r"$\sigma$", r"$\lambda_0$", r"$\lambda_1$"],
)

iterations = [j for i in range(dimp) for j in range(dimi)]
participants = [j for j in range(dimi) for i in range(dimp)]
ax = seaborn.pairplot(df, kind="reg", diag_kind="kde")
plt.ion()
ax.fig.savefig(fname="img/pairplot_emg_jgp.pdf")
plt.show()


df["iteration"] = iterations
df["participant"] = participants
