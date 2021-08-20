import numpy as np
import matplotlib.pyplot as plt
with open("./logistic_cpp.log") as f:
    l=list(map(float,f.read().split()))
a=l[::2]
x=l[1::2]
fig=plt.figure(figsize=[6.4,4.8])
ax=fig.add_subplot(1,1,1)
ax.plot(a,x,"cs",markersize=0.01)
ax.set_xlabel("a")
ax.set_ylabel("x")
plt.savefig("./graph_cpp.png",dpi=800)
