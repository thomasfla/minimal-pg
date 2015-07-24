import numpy as np
import matplotlib.pyplot as plt
Tds=0.5
g=9.81
h=0.63
w2= g/h
w = np.sqrt(w2)



def Dx(T):
    return np.matrix([[np.cosh(w*T)   ,np.sinh(w*T)/w],
                       [np.sinh(w*T)*w ,np.cosh(w*T)  ]])   
                       
def Du(T):
    return np.matrix([[ 1-np.cosh(w*T)-T/Tds  , T/Tds ],
                       [-w*np.sinh(w*T)-T/Tds  ,-T/Tds ]])                  
                       
                       
def Sx(T):
    return np.matrix([[np.cosh(w*T)   ,np.sinh(w*T)/w],
                       [np.sinh(w*T)*w ,np.cosh(w*T)  ]])   
                       
def Su(T):
    return np.matrix([[ 1-np.cosh(w*T)  ,0 ],
                       [-w*np.sinh(w*T)  ,0 ]])     

cc=[]
p0=-0.0
p1=1.0
p01=np.matrix([[p0],[p1]])
c0=0.05
dc0=-0.5
x0=np.matrix([[c0],[dc0]])
tt=np.linspace(0,Tds,100)
for t in tt:
    x=Dx(t)*x0+Du(t)*p01
    cc.append(x[0,0])
plt.plot(tt,cc)
plt.hold(True)
plt.plot([0,Tds],[p0,p1])
plt.show()
