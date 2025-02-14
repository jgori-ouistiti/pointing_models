import matplotlib.pyplot as plt
import numpy
from matplotlib.patches import Ellipse

import matplotlib.patches as mpatches
import matplotlib.path as mpath

plt.style.use("fivethirtyeight")


def draw_ellipse(mean, cov, ax, edgecolor="b", facecolor="b"):
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
    )
    ax.add_patch(ellipse)


fig, ax = plt.subplots(1, 1)


Path = mpath.Path
path_data = [
    (Path.MOVETO, (0, -0.5)),
    (Path.CURVE3, (-0.5, 0)),
    (Path.CURVE3, (0.5, 0.75)),
    (Path.LINETO, (4, 3.5)),
    (Path.CURVE3, (4.5, 1.5)),
    (Path.CURVE3, (4, 0)),
    (Path.CLOSEPOLY, (0, -0.5)),
]

codes, verts = zip(*path_data)
path = mpath.Path(verts, codes)
patch = mpatches.PathPatch(path, facecolor="r", alpha=0.5)
ax.add_patch(patch)

x, y = zip(*path.vertices)
(line,) = ax.plot(x, y, "go-")

for i in range(4):
    mean = numpy.array([i, i / 2])
    cov = numpy.array([[1 / 100, 0], [0, (1 + 4 * i) / 100]])
    draw_ellipse(mean, cov, ax)

# ax.set_xlim([-1, 4])
# ax.set_ylim([-1, 3])
# ax.set_xticklabels([])
# ax.set_yticklabels([])
ax.set_xlabel("ID (arbitrary units)")
ax.set_ylabel("MT (arbitrary units)")
plt.ion()
plt.tight_layout()
plt.show()
