import numpy as np
from IPython import embed
import matplotlib.pyplot as plt
tt=np.linspace(0,1,1000)
cc=[]
threshold =0.6
for ev in tt:
    if ev > threshold:
        c= 1/(1-ev+0.01) - 1/(1-threshold+0.01)
    else:
        c=0.0
    cc.append(c)
plt.plot(tt,cc)

plt.show()
