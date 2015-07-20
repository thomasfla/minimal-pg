def foot_interpolate(x0,y0,x1,y1,ev):
    dtotal = np.sqrt((x1-x0)**2 + (y1-y0)**2)
    alpha = np.arctan2((x1-x0),(y1-y0))
    r=0.05
    if (2*r)>dtotal: # in small step, do not go to max height
        r=dtotal/2
    d2=dtotal-2*r
    #p=(ev)*(np.pi+d2/r)/T#corresponding parameter in parametric curve
    p=(np.pi+d2/r)*(1-np.cos( ev*np.pi))/2 #idem but smoother (v is C0)
    if (p >= 0         and p < np.pi/2     ):
        x=(1-np.cos(p))*r
        z=(np.sin(p))*r
    elif (p>=np.pi/2   and p < np.pi/2+d2/r):
        x=(1+p-(np.pi/2))*r
        z=r
    elif (np.pi/2+d2/r and p < d2/r+np.pi  ):
        x=d2+(np.sin(p-(np.pi/2+d2/r))+1)*r
        z=np.cos(p-(np.pi/2+d2/r))*r
    elif (p < 0):
        return (np.array([x0,y0,0]))
    elif (p >= d2/r+np.pi):
        return (np.array([x1,y1,0]))            
    xo=x0+x*np.sin(alpha) #orient the trajectory #offset x0,y0
    yo=y0+x*np.cos(alpha)
    return [xo,yo,z]  

#!/usr/bin/env python
    #Minimal PG using only steps positions as parameters
    #by using analytic solve of LIP dynamic equation
from pinocchio_controller import PinocchioController
from minimal_pg import PgMini
import matplotlib.pyplot as plt
import numpy as np
import time
print ("start")
#define const
Nstep=8
g=9.81
h=0.63
durrationOfStep=0.5
Dpy=0.20
beta_x=3.0
beta_y=10.0

USE_WIIMOTE=False
USE_GAMEPAD=True
DISPLAY_PREVIEW=False
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
p=PinocchioController()
pps=20 #point per step
v=[1.0,0.1]
p0=[-0.01,-0.01]
x0=[[0,0] , [0,0]]
comx=[]
comy=[]

lastFoot=p0

LR=True
#plt.ion()
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
            v[0]=-my_joystick.get_axis(1)
            v[1]=-my_joystick.get_axis(0)
        steps = pg.computeStepsPosition(ev,p0,v,x,LR)
        currentFoot = [steps[0][0],steps[1][0]]
        nextFoot    = [steps[0][1],steps[1][1]]
        
        if DISPLAY_PREVIEW:
            [tt, cc_x , cc_y , d_cc_x , d_cc_y] = pg.computePreviewOfCom(steps,ev,x,N=2)
        #plot data
        #plt.axis((-1,5,-1,1))
        #plt.plot(cc_x,cc_y,'g',lw=0.5)
        #plt.hold(True)
        #plt.plot(steps[0],steps[1],'rD')
        #plt.plot([steps[0][0]],[steps[1][0]],'bD')
        
        #plt.plot([c_x],[c_y],"D")
        #plt.plot(comx,comy,"k")
        #plt.plot([steps[0][1]],[steps[1][1]],"Dy")

        [xf,yf,zf] = foot_interpolate(lastFoot[0],lastFoot[1],nextFoot[0],nextFoot[1],ev)
        #plt.plot([xf],[yf],"Dg")
        
        if LR :
            left_foot_xyz  = [xf,yf,zf]
            right_foot_xyz = [steps[0][0],steps[1][0],0.0] #current support foot
        else :
            right_foot_xyz = [xf,yf,zf]
            left_foot_xyz  = [steps[0][0],steps[1][0],0.0] #current support foot
        
        left_foot=left_foot_xyz[:2]
        right_foot=right_foot_xyz[:2]
        #plt.plot([left_foot[0] , x[0][0]],[left_foot[1]  ,x[1][0]],"r",lw=10)
        #plt.plot([right_foot[0], x[0][0]],[right_foot[1] ,x[1][0]],"g",lw=10)
        #plt.draw()  
        FlagRT = False
        while(time.time()-t0 < (durrationOfStep/pps)):
            FlagRT = True
        if not FlagRT :
            print "not in real time !"

        p.controlLfRfCom(left_foot_xyz,right_foot_xyz,[x[0][0],x[1][0],h],1.5)
        t0=time.time()
        #plt.clf()
    #prepare next point
    lastFoot = currentFoot
    p0 = nextFoot 
    x0 = x
    LR = not LR
    
if USE_WIIMOTE:
    wm.close()
