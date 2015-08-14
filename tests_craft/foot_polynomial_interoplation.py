from IPython import embed
import matplotlib.pyplot as plt
import numpy as np



#condition initiales:



#
a= -2
b= 3
c= 0
d= 0

#plot
tt=[]
ff=[]
for t in np.linspace(0,1,100):
    f=a*t**3 + b*t**2 + c*t + d
    tt.append (t)
    ff.append (f)
    
    
plt.plot(tt,ff)
plt.show()
