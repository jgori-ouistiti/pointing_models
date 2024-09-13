import matplotlib.pyplot as plt
import os
import pandas
import seaborn

plt.style.use("fivethirtyeight")

df_agg = pandas.DataFrame(
    columns=[
        "Unnamed: 0",
        "participant",
        "day",
        "iteration",
        "A",
        "W",
        "ID",
        "IDe(2d)",
        "Duration",
    ]
)
for _file in os.listdir("exchange/jgp"):
    if _file.split("_")[0] != "0":
        continue
    df = pandas.read_csv(f"exchange/jgp/{_file}")
    df_agg = pandas.concat([df_agg, df])

mt = df_agg["Duration"]
ide = df_agg["IDe(2d)"]

from scipy.stats import exponnorm, uniform
import numpy

# mt
emg_marginal_fit = exponnorm.fit(mt)
emg_x = numpy.linspace(numpy.min(mt), numpy.max(mt), 100)
emg_y = [exponnorm.pdf(__x, *emg_marginal_fit) for __x in emg_x]

# ide
unif_marginal_fit = uniform.fit(ide)
unif_x = numpy.linspace(numpy.min(ide), numpy.max(ide), 100)
unif_y = [uniform.pdf(__x, *unif_marginal_fit) for __x in unif_x]


fig, axs = plt.subplots(2, 2)


seaborn.histplot(
    x=mt, kde=True, ax=axs[0, 0], stat="density", line_kws={"label": "KDE fit"}
)
axs[0, 0].plot(emg_x, emg_y, "r-", label="EMG fit")
axs[0, 0].set_xlabel("MT (s)")
axs[0, 0].legend()
K, loc, scale = emg_marginal_fit
axs[0, 0].set_title(
    rf"$\beta$={loc:.2f}, $\sigma$={scale:.2f}, $\lambda$={K*scale:.2f}"
)

seaborn.histplot(
    x=ide, kde=True, ax=axs[0, 1], stat="density", line_kws={"label": "KDE fit"}
)
axs[0, 1].plot(unif_x, unif_y, "r-", label="EMG fit")
axs[0, 1].set_xlabel("IDe (bit)")
axs[0, 1].legend()
loc, scale = unif_marginal_fit
axs[0, 1].set_title(rf"min={loc:.2f}, max={loc+scale:.2f}")

axs[1, 0].plot(ide, mt, "o", label="Pointing data")
axs[1, 0].set_xlabel("IDe (bit)")
axs[1, 0].set_ylabel("MT (s)")
plt.ion()
plt.tight_layout()
plt.show()
