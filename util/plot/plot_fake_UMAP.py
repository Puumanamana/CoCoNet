import numpy as np
import matplotlib.pyplot as plt

N = 20
col = ['r','g','b']
centers = np.array([ [0.4,0],[0.5,0.2],[1,0.8] ])
points = [ np.array([np.random.normal(x,1,N), np.random.normal(y,1,N)]) for x,y in centers ]

fig,ax = plt.subplots()
for i, (center,pts) in enumerate(zip(centers,points)):
    ax.scatter(pts[0],pts[1],s=20,c=col[i],label="Contig_{}".format(i+1))
    circle = plt.Circle(center, 0.15, color=col[i], alpha=0.5, clip_on=False)
    ax.add_artist(circle)
    if i == 0:
        med_dist = np.median(np.sqrt(np.sum((pts-center[:,None])**2,axis=0)))
        area = plt.Circle(center, med_dist, color=col[i], fill=False, clip_on=False)
        ax.add_artist(area)
plt.legend()
plt.show()
