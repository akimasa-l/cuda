import numpy as np
import matplotlib.pyplot as plt
with open("./logistic.log") as f:
    l=list(map(float,f.read().split()))
a=l[::2]
x=l[1::2]
fig=plt.figure(figsize=[6.4,4.8])
ax=fig.add_subplot(1,1,1)
ax.plot(a,x,"c.",markersize=.01)
ax.set_xlabel("a")
ax.set_ylabel("x")
plt.savefig("./graph.png",dpi=800)
