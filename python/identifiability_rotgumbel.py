import seaborn
import pandas
import matplotlib.pyplot as plt
import numpy

plt.style.use(["fivethirtyeight"])

df = pandas.read_csv("../R/exchange/identifiability_joint.csv")
df.columns = [50, 100, 200, 500, 1000]
long_df = pandas.melt(df, var_name="N", value_name="R")
long_df["logR"] = numpy.log(long_df["R"])

df_bootstrap = pandas.read_csv("../R/exchange/identifiability_bootstrap.csv")
df_bootstrap.columns = [50, 100, 200, 500, 1000]
long_df_bootstrap = pandas.melt(df_bootstrap, var_name="N", value_name="R")
long_df_bootstrap["logR"] = numpy.log(long_df_bootstrap["R"])

seaborn.regplot(long_df, x="N", y="logR", fit_reg=True, logx=True, label="continuous")
seaborn.regplot(
    long_df_bootstrap, x="N", y="logR", fit_reg=True, logx=True, label="blocks"
)
plt.legend()
plt.tight_layout()
plt.ion()
plt.savefig("img/identifiability.pdf")
plt.show()
