import seaborn
import matplotlib.pyplot as plt
import pickle
import numpy
from matplotlib.colors import LinearSegmentedColormap


with open("gop_emg_params_all_p_s_r.pkl", "rb") as _file:
    emg = pickle.load(_file)

emg = emg.to_pandas()
fig, axs = plt.subplots(1, 5)
seaborn.scatterplot(emg, x="Strategy", y="beta_0", ax=axs[0])
seaborn.scatterplot(emg, x="Strategy", y="beta_1", ax=axs[1])
seaborn.scatterplot(emg, x="Strategy", y="sigma", ax=axs[2])
seaborn.scatterplot(emg, x="Strategy", y="lambda_0", ax=axs[3])
seaborn.scatterplot(emg, x="Strategy", y="lambda_1", ax=axs[4])

axs[0].plot([i for i in range(1, 6)], emg.groupby("Strategy").mean()["beta_0"], "Dr")
axs[1].plot([i for i in range(1, 6)], emg.groupby("Strategy").mean()["beta_1"], "Dr")
axs[2].plot([i for i in range(1, 6)], emg.groupby("Strategy").mean()["sigma"], "Dr")
axs[3].plot([i for i in range(1, 6)], emg.groupby("Strategy").mean()["lambda_0"], "Dr")
axs[4].plot([i for i in range(1, 6)], emg.groupby("Strategy").mean()["lambda_1"], "Dr")


plt.ion()
plt.show()
