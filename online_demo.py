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
Dpy=0.20
beta_x=1.0
beta_y=5.0

USE_WIIMOTE=False
USE_GAMEPAD=True

if USE_WIIMOTE:
    import cwiid
    print "Wiimote : press 1+2"
    wm = cwiid.Wiimote()
    time.sleep(0.5)
    wm.led=1
    wm.rpt_mode = cwiid.RPT_BTN | cwiid.RPT_ACC #Btns and Accelerometer
    
if USE_GAMEPAD:
    import pygame
    import time
    pygame.init()
    pygame.joystick.init()
    pygame.event.pump()
    if (pygame.joystick.get_count() > 0):
        print "found gamepad! : " + pygame.joystick.Joystick(0).get_name()
        my_joystick = pygame.joystick.Joystick(0)
        my_joystick.init()
    else :
        print "No gamepad found"
        USE_GAMEPAD = False


sigmaNoisePosition=0.0
sigmaNoiseVelocity=0.00
#initialisation of the pg
pg = PgMini(Nstep,g,h,durrationOfStep,Dpy,beta_x,beta_y)     

pps=5 #point per step
v=[1.0,0.1]
p0=[-0.01,-0.01]
x0=[[0,0] , [0,0]]
comx=[]
comy=[]

LR=True
plt.ion()
t0=time.time()
for k in range (80): #do 80 steps
    for ev in np.linspace(1.0/pps,1,pps):
        t=durrationOfStep*ev
        [c_x , c_y , d_c_x , d_c_y]         = pg.computeNextCom(p0,x0,t)
        x=[[c_x,d_c_x] , [c_y,d_c_y]]
        if sigmaNoisePosition >0:     
            x[0][0]+=np.random.normal(0,sigmaNoisePosition) #add some disturbance!
            x[1][0]+=np.random.normal(0,sigmaNoisePosition)
        if sigmaNoiseVelocity >0:  
            x[0][1]+=np.random.normal(0,sigmaNoiseVelocity)
            x[1][1]+=np.random.normal(0,sigmaNoiseVelocity)

        comx.append(x[0][0])
        comy.append(x[1][0])
        
        if USE_WIIMOTE:
            v[0]=v[0]*0.2 + 0.8*(wm.state['acc'][0]-128)/50.0
            v[1]=v[1]*0.2 + 0.8*(wm.state['acc'][1]-128)/50.0    
        if USE_GAMEPAD:
            pygame.event.pump()
            v[0]=my_joystick.get_axis(0)
            v[1]=-my_joystick.get_axis(1)
        print x
        steps = pg.computeStepsPosition(ev,p0,v,x,LR)
        [tt, cc_x , cc_y , d_cc_x , d_cc_y] = pg.computePreviewOfCom(steps,ev,x,N=20)
        #plot data
        plt.axis((-1,5,-1,1))
        plt.plot(cc_x,cc_y,'g',lw=0.5)
        plt.hold(True)
        plt.plot(steps[0],steps[1],'rD')
        plt.plot([steps[0][0]],[steps[1][0]],'bD')
        
        plt.plot([c_x],[c_y],"D")
        plt.plot(comx,comy,"k")
        
        plt.draw()  
        FlagRT = False
        while(time.time()-t0 < (durrationOfStep/pps)):
            FlagRT = True
        if not FlagRT :
            print "not in real time !"
        t0=time.time()
        plt.clf()
        #prepare next point
    p0 = [steps[0][1],steps[1][1]] #
    x0 = x
    LR = not LR
    plt.show()

if USE_WIIMOTE:
    wm.close()
