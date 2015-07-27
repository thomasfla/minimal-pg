import numpy as np
import matplotlib.pyplot as plt
Tds=0.1
Tss=0.6

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
    return np.matrix([[ 1-np.cosh(w*T)+ (1/Tds)*((1/w)*np.sinh(w*T) -T)  ,(1/Tds) * (T-(1/w)*np.sinh(w*T))],
                      [-w*np.sinh(w*T)- (1/Tds)*(np.cosh(w*T)-1)        , (1/Tds) * (1-np.cosh(w*T)) ]])                  
                       
def Sx(T):
    return np.matrix([[np.cosh(w*T)   ,np.sinh(w*T)/w],
                       [np.sinh(w*T)*w ,np.cosh(w*T)  ]])   
                       
def Su(T):
    return np.matrix([[ 1-np.cosh(w*T)  ,0 ],
                       [-w*np.sinh(w*T)  ,0 ]])     
eps=1e-2

cc=[]
p0=-0.223
p1=0.2
p2=-0.2
p3=0.2
p4=-0.2
p5=0.2
c0=0.05
dc0=0.0
x0=np.matrix([[c0],[dc0]])

dcc=[]
ddcc=[]
aa=[]
tt=[]
pp=[]

p01=np.matrix([[p0],[p1]])
ttDS=np.linspace(0,Tds,1000) #double support
for t in ttDS:
    x=Dx(t)*x0+Du(t)*p01
    cc.append(x[0,0])
    dcc.append(x[1,0])
    p=p01[0,0]+t*(p01[1,0]-p01[0,0])/Tds
    pp.append(p)
    aa.append(w2*(x[0,0]-p))
    tt.append(t)

p01=np.matrix([[p1],[p1]])
x0=x
ttSs=np.linspace(0,Tss,1000) #Simple support
for t in ttSs:
    x=Sx(t)*x0+Su(t)*p01
    cc.append(x[0,0])
    dcc.append(x[1,0])
    p=p01[0,0]+t*(p01[1,0]-p01[0,0])/Tss
    pp.append(p)
    aa.append(w2*(x[0,0]-p))
    tt.append(t+Tds)   
p01=np.matrix([[p1],[p2]])
x0=x
ttDS=np.linspace(0,Tds,1000) #double support
for t in ttDS:
    x=Dx(t)*x0+Du(t)*p01
    cc.append(x[0,0])
    dcc.append(x[1,0])
    p=p01[0,0]+t*(p01[1,0]-p01[0,0])/Tds
    pp.append(p)
    aa.append(w2*(x[0,0]-p))
    tt.append(t+Tds+Tss)
 
 

p01=np.matrix([[p2],[p2]])
x0=x
ttSs=np.linspace(0,Tss,1000) #Simple support
for t in ttSs:
    x=Sx(t)*x0+Su(t)*p01
    cc.append(x[0,0])
    dcc.append(x[1,0])
    p=p01[0,0]+t*(p01[1,0]-p01[0,0])/Tss
    pp.append(p)
    aa.append(w2*(x[0,0]-p))
    tt.append(t+2*Tds+Tss)
x0=x
p01=np.matrix([[p2],[p3]])
ttDS=np.linspace(0,Tds,1000) #double support
for t in ttDS:
    x=Dx(t)*x0+Du(t)*p01
    cc.append(x[0,0])
    dcc.append(x[1,0])
    p=p01[0,0]+t*(p01[1,0]-p01[0,0])/Tds
    pp.append(p)
    aa.append(w2*(x[0,0]-p))
    tt.append(t+2*Tds+2*Tss)


p01=np.matrix([[p3],[p3]])
x0=x
ttSs=np.linspace(0,Tss,1000) #Simple support
for t in ttSs:
    x=Sx(t)*x0+Su(t)*p01
    cc.append(x[0,0])
    dcc.append(x[1,0])
    p=p01[0,0]+t*(p01[1,0]-p01[0,0])/Tss
    pp.append(p)
    aa.append(w2*(x[0,0]-p))
    tt.append(t+3*Tds+2*Tss)
    
p01=np.matrix([[p3],[p4]])
x0=x
ttDS=np.linspace(0,Tds,1000) #double support
for t in ttDS:
    x=Dx(t)*x0+Du(t)*p01
    cc.append(x[0,0])
    dcc.append(x[1,0])
    p=p01[0,0]+t*(p01[1,0]-p01[0,0])/Tds
    pp.append(p)
    aa.append(w2*(x[0,0]-p))
    tt.append(t+3*Tds+3*Tss)
    
tmp=0
ddcc=[]
for dc in dcc:
    ddcc.append( (dc-tmp)/(tt[1]-tt[0]) )
    tmp=dc
    
plt.plot(tt,cc,'b')
plt.plot(tt,dcc,'g')
plt.plot(tt,ddcc,'y')
plt.plot(tt,aa,'r')
plt.hold(True)
plt.plot(tt,pp,'k')
#plt.plot([0,Tds,Tds+Tss,2*Tds+Tss,2*Tds+2*Tss,3*Tds+2*Tss,3*Tds+3*Tss,4*Tds+3*Tss],[p0,p1,p1,p2,p2,p3,p3,p4])
plt.show()
