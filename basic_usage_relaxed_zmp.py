#!/usr/bin/env python



import matplotlib.pyplot as plt
import numpy as np
import time
from minimal_pg_relaxed_zmp import PgMini
#initialisation of the pg
pg = PgMini()               

#solve and return steps placement
#~ t0=time.time()  #(tic tac mesurement)
#~ steps = pg.computeStepsPosition() 
#~ print "compute time: " + str((time.time()-t0)*1e3)  + " milliseconds"


#~ for ery in np.linspace(-0.01,0.01,3):
    #~ for erx in np.linspace(-0.01,0.01,3):
        #~ x0err=[[0+erx,0+ery] , [0,0]]
        #~ #get COM at a particular time value
        #~ steps = pg.computeStepsPosition(alpha=0.0,p0=[-0.001,-0.005],v=[1.0,0.1],x0=x0err,LR=True) 
        #~ #get the COM preview
        #~ [tt, cc_x , cc_y , d_cc_x , d_cc_y] = pg.computePreviewOfCom(steps,alpha=0.0,x0=x0err,N=20)
        #~ plt.hold(True)
        #~ plt.plot(steps[0],steps[1])
        #~ plt.plot(cc_x,cc_y)
        #~ 

p0=[0.0,0.0]
v=[1.0,0.1]
alpha=0.0
x0=[[0.0,0.1] , [0.0,0.1]]
dt=pg.durrationOfStep / 30.0
N=pg.Nstep
LR=True

log_zmp_x=[]
log_zmp_y=[]

for k in range(5): #number of steps to do:
    for i in range(30): #evolution in the current step
        plt.figure(1)
        plt.plot([p0[0]],[p0[1]],"dr")
        plt.annotate('p0*',xy=(p0[0],p0[1]), xytext = (0, 0), textcoords = 'offset points')
        plt.plot([x0[0][0]],[x0[1][0]],'Db')
        
        
        steps = pg.computeStepsPosition(alpha,p0,v,x0,LR)
        #~ print p0[0]-steps[0][0]
        #~ print p0[1]-steps[1][0]
        plt.hold(True)
        #~ plt.plot(steps[0],steps[1],"d")
        
        log_zmp_x.append(steps[0][0])
        log_zmp_y.append(steps[1][0])
        
        labels = ['p{0} - t{1}'.format(k,i) for k in range(N)]
        #~ for label, x, y in zip(labels, steps[0], steps[1]):
            #~ plt.annotate(
                #~ label, 
                #~ xy = (x, y), xytext = (-20, 20),
                #~ textcoords = 'offset points', ha = 'right', va = 'bottom',
                #~ bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
                #~ arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

        [tt, cc_x , cc_y , d_cc_x , d_cc_y] = pg.computePreviewOfCom(steps,alpha,x0,111)
        

        #~ print ('len =')
        #~ print len(tt)
        #~ plt.plot(cc_x,cc_y)
        
        #~ plt.figure(2)
        #~ plt.plot(tt,d_cc_y,'-x')
        #~ plt.plot([alpha*pg.durrationOfStep,alpha*pg.durrationOfStep],[-1,1])
        
        #prepare next iteration
        zmp=[steps[0][0],steps[1][0]]
        tmp=pg.computeNextCom(zmp,x0,dt)
        x0=[  [ tmp[0],tmp[2] ] , [tmp[1],tmp[3] ]  ]
        alpha=alpha+1/30.0
    #ends the current step:
    alpha=0.0
    p0=[steps[0][1],steps[1][1]]
    LR=not LR
    
plt.plot(log_zmp_x,log_zmp_y,'-D')
plt.show()
