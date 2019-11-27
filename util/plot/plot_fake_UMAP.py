import sys
import numpy as np
import matplotlib.pyplot as plt

nb = int(sys.argv[1])
N = 30
col = ['#DB8A63', '#A086AD', '#9AB6CA']
centers = [np.array([[0.2, 0.1], [-0.2, -0.1], [1, -0.2]]),
           np.array([[0.2, 0.1], [0.5, -0.2], [-0.2, -0.1]])][nb]
points = [np.array([np.random.normal(x, 0.5, N), np.random.normal(y, 0.5, N)]) for x, y in centers]
triangle_dims = np.array([[0, .4], [-0.4, -0.4], [0.4, -0.4]])*.15

fig, ax = plt.subplots(figsize=(10, 10))
for i, (center, pts) in enumerate(zip(centers, points)):
    ax.scatter(pts[0], pts[1], s=100, c=col[i], label="Contig_{}".format(i+1), alpha=0.85)
    # circle = plt.Circle(center, 0.15, color=col[i], alpha=0.5, clip_on=False)
    triangle = plt.Polygon(triangle_dims+center, color=col[i], alpha=1, clip_on=False)
    ax.add_artist(triangle)
    # if i == 0:
    #     radius = 2*np.std(np.sqrt(np.sum((pts-center[:, None])**2, axis=0)))
    #     area = plt.Circle(center, radius, color=col[i], fill=False, clip_on=False)
    #     ax.add_artist(area)
    ax.set_xticks([])
    ax.set_yticks([])

for spine in ax.spines.values():
    spine.set_visible(False)

plt.savefig(f"/home/cedric/repr_{nb}.pdf", transparent=True)
plt.show()
