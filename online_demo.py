#!/usr/bin/env python
    #Minimal PG using only steps positions as parameters
    #by using analytic solve of LIP dynamic equation

from minimal_pg import PgMini
import matplotlib.pyplot as plt
import numpy as np
import time
print ("start")
#define const
Nstep=8
g=9.81
h=0.63
durrationOfStep=0.7
Dpy=0.30
beta_x=1.5
beta_y=5.0

noizeSigma=0.005

#initialisation of the pg
pg = PgMini(Nstep,g,h,durrationOfStep,Dpy,beta_x,beta_y)     


v=[1.0,0.1]
p0=[-0.01,-0.01]
x0=[[0,0] , [0,0]]
comx=[]
comy=[]

LR=True
plt.ion()
for k in range (10): #do 20 steps
    for ev in np.linspace(0,1,10):
        t=durrationOfStep*ev
        [c_x , c_y , d_c_x , d_c_y]         = pg.computeNextCom(p0,x0,t)
        x=[[c_x,d_c_x] , [c_y,d_c_y]]
        if noizeSigma >0:     
            x[0][0]+=np.random.normal(0,noizeSigma) #add some disturbance!
            x[0][1]+=np.random.normal(0,noizeSigma)
            x[1][0]+=np.random.normal(0,noizeSigma)
            x[1][1]+=np.random.normal(0,noizeSigma)
        comx.append(x[0][0])
        comy.append(x[1][0])
        steps = pg.computeStepsPosition(ev,p0,v,x,LR)
        [tt, cc_x , cc_y , d_cc_x , d_cc_y] = pg.computePreviewOfCom(steps,ev,x,N=20)
        #plot data
        plt.axis((-1,5,-1,1))
        plt.plot(cc_x,cc_y,'b')
        plt.hold(True)
        plt.plot(steps[0],steps[1],'r')
        plt.plot([c_x],[c_y],"D")
        plt.plot(comx,comy,"k")
        
        plt.draw()
        time.sleep(0.1)
        #prepare next point
    p0 = [steps[0][1],steps[1][1]] #
    x0 = x
    LR = not LR
    plt.clf()

plt.show()
