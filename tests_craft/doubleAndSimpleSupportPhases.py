import numpy as np
import matplotlib.pyplot as plt
Tds=0.1
Tss=0.7

g=9.81
h=0.63
w2= g/h
w = np.sqrt(w2)

def Dx(T):
    return np.matrix([[np.cosh(w*T)   ,np.sinh(w*T)/w],
                       [np.sinh(w*T)*w ,np.cosh(w*T)  ]])   
                       
#~ def Du(T):
    #~ return np.matrix([[ 1-np.cosh(w*T)-T/Tds  , T/Tds ],
                       #~ [-w*np.sinh(w*T)-1/Tds  , 1/Tds ]])                  
                       
      

def Du(T):
    return np.matrix([[ 1-np.cosh(w*T)+ (1/Tds)*((1/w)*np.sinh(w*T) -T)  ,(1/Tds) * (T-(1/w)*np.sinh(w*T))],[-w*np.sinh(w*T)- (1/Tds)*(np.cosh(w*T)-1)        ,  1/Tds*(1-np.cosh(w*T)) ]])                  
                       
def Sx(T):
    return np.matrix([[np.cosh(w*T)   ,np.sinh(w*T)/w],
                       [np.sinh(w*T)*w ,np.cosh(w*T)  ]])   
                       
def Su(T):
    return np.matrix([[ 1-np.cosh(w*T)  ,0 ],
                       [-w*np.sinh(w*T)  ,0 ]])     
eps=1e-5

cc=[]
p0=-0.2
p1=0.2
p2=-0.2
p3=0.2
c0=-0.1
dc0=0.0
x0=np.matrix([[c0],[dc0]])

dcc=[]
aa=[]
tt=[]

p01=np.matrix([[p0],[p1]])
ttDS=np.linspace(0,Tds,1000) #double support
for t in ttDS:
    x=Dx(t)*x0+Du(t)*p01
    cc.append(x[0,0])
    dcc.append(x[1,0])
    tt.append(t)

p01=np.matrix([[p1],[p1]])
x0=x
ttSs=np.linspace(0,Tss,1000) #Simple support
for t in ttSs:
    x=Sx(t)*x0+Su(t)*p01
    cc.append(x[0,0])
    dcc.append(x[1,0])
    tt.append(t+Tds)   
    
p01=np.matrix([[p1],[p2]])
x0=x
ttDS=np.linspace(0,Tds,1000) #double support
for t in ttDS:
    print x
    x=Dx(t)*x0+Du(t)*p01
    cc.append(x[0,0])
    dcc.append(x[1,0])
    tt.append(t+Tds+Tss)
    
plt.plot(tt,cc)
#plt.plot(tt,dcc)
plt.hold(True)
plt.plot([0,Tds,Tds+Tss,2*Tds+Tss],[p0,p1,p1,p2])
plt.show()
