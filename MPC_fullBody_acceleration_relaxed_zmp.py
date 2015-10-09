#!/usr/bin/env python
    #Minimal PG using only steps positions as parameters
    #by using analytic solve of LIP dynamic equation
    
import pinocchio_controller_acceleration
reload(pinocchio_controller_acceleration) #reload in case file has changed
from pinocchio_controller_acceleration import PinocchioControllerAcceleration
import pinocchio as se3
from IPython import embed
from minimal_pg_relaxed_zmp import PgMini
#~ from minimal_pg import PgMini
import matplotlib.pyplot as plt
import numpy as np
import time
print ("start")

#define const
Nstep=3
pps=100 #point per step
g=9.81
h=0.63
durrationOfStep=0.4
Dpy=0.20

beta_x=3.0 
beta_y=8.0

N_COM_TO_DISPLAY = 10

USE_WIIMOTE=False
USE_GAMEPAD=True
DISPLAY_PREVIEW=False
ENABLE_LOGING=False
STOP_TIME = np.inf

sigmaNoisePosition=0.00
sigmaNoiseVelocity=0.00
#initialisation of the pg

dt=durrationOfStep/pps
print( "dt= "+str(dt*1000)+"ms")
pg = PgMini(Nstep,g,h,durrationOfStep,Dpy,beta_x,beta_y)     
p=PinocchioControllerAcceleration(dt)

v=[1.0,1.0]
#initial feet positions
p0      =[0.0102606,-0.096]
cop=p0
lastFoot=[0.0102606,0.096]

#p0=[0.0102606,    -0.01]
 #0.0102606    -0.096 0.0669995
 #~ #0.0102606     0.096 0.0669995    

def prepareCapsForStepPreviewInViewer (robot):
    for i in range(Nstep):
        if i == 0:
            robot.viewer.gui.addSphere("world/pinocchio/capsSteps"+str(i),0.05,[1,0,0,0.5])
        else:
            robot.viewer.gui.addSphere("world/pinocchio/capsSteps"+str(i),0.05,[0,1,0,0.5])

def prepareCapsForComPreviewInViewer (robot):
    for i in range(N_COM_TO_DISPLAY*Nstep):
        robot.viewer.gui.addSphere("world/pinocchio/capsCom"+str(i),0.01,[1,0,0,0.5])

def showStepPreviewInViewer (robot,steps):
    for i in range(Nstep):
        XYZ_caps=np.matrix([[steps[0][i]],[steps[1][i]],[.0]])
        RPY_caps=np.matrix([[.0],[.0],[.0]])
        SE3_caps = se3.SE3(se3.utils.rpyToMatrix(RPY_caps),XYZ_caps)
        robot.viewer.gui.applyConfiguration("world/pinocchio/capsSteps"+str(i),se3.utils.se3ToXYZQUAT(SE3_caps))
        
def showComPreviewInViewer (robot,COMs):
    for i in range(len(COMs[0])):
        XYZ_caps=np.matrix([[COMs[0][i]],[COMs[1][i]],[.0]])
        RPY_caps=np.matrix([[.0],[.0],[.0]])
        SE3_caps = se3.SE3(se3.utils.rpyToMatrix(RPY_caps),XYZ_caps)
        robot.viewer.gui.applyConfiguration("world/pinocchio/capsCom"+str(i),se3.utils.se3ToXYZQUAT(SE3_caps))    

def foot_interpolate(x0,y0,x1,y1,ev,durrationOfStep=1.0):
    dtotal = np.sqrt((x1-x0)**2 + (y1-y0)**2)
    alpha = np.arctan2((x1-x0),(y1-y0))
    r=0.05
    if (2*r)>dtotal: # in small step, do not go to max height
        r=dtotal/2
    d2=dtotal-2*r
    #p=(ev)*(np.pi+d2/r)/T#corresponding parameter in parametric curve
    p=(np.pi+d2/r)*(1-np.cos( ev*np.pi))/2 #idem but smoother (v is C0)
    dp=(np.pi/2)*(d2/r+np.pi)*np.sin(np.pi*ev)/durrationOfStep
    if (p >= 0         and p < np.pi/2     ):
        x=(1-np.cos(p))*r
        z=(np.sin(p))*r
        dx=np.sin(p)*r*dp
        dz=np.cos(p)*r*dp 
    elif (p>=np.pi/2   and p < np.pi/2+d2/r):
        x=(1+p-(np.pi/2))*r
        z=r
        dx=r*dp 
        dz=0*dp 
    elif (np.pi/2+d2/r and p < d2/r+np.pi  ):
        x=d2+(np.sin(p-(np.pi/2+d2/r))+1)*r
        z=np.cos(p-(np.pi/2+d2/r))*r
        dx=-np.sin(d2/r - p)*r*dp
        dz= np.cos(d2/r - p)*r*dp
    elif (p < 0):
        return (np.array([x0,y0,0, 0,0,0]))
    elif (p >= d2/r+np.pi):
        return (np.array([x1,y1,0 ,0,0,0]))            
    xo=x0+x*np.sin(alpha) #orient the trajectory #offset x0,y0
    yo=y0+x*np.cos(alpha)
    
    dxo=dx*np.sin(alpha)
    dyo=dx*np.cos(alpha)

    return [xo,yo,z,dxo,dyo,dz]

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
prepareCapsForStepPreviewInViewer(p.robot)
prepareCapsForComPreviewInViewer(p.robot)
initial_com=p.robot.com(p.robot.q0)
x0=[[initial_com[0,0],initial_com[1,0]] , [0.0,0.0]]
x=x0
comx=[]
comy=[]

vect_f=[]
vect_df=[]
LR=True
#plt.ion()
t0=time.time()
simulationTime=0.0

debug_com_1=[]
debug_com_2=[]

if ENABLE_LOGING:
    log_comx_mesure=[]
    log_comx_cmd=[]
    log_t=[]
RUN_FLAG=True
#~ ev=1.0-1.0/pps
#ev=0.0*pps*(1.0/pps) #start at 80 %
ev=0.0
tk=0 
while(RUN_FLAG):
    #~ for ev in np.linspace(1.0/pps,1,pps):
    while(ev<1.0 and RUN_FLAG):
        #~ time.sleep(0.3)
        t=durrationOfStep*ev

        [c_x , c_y , d_c_x , d_c_y]     = pg.computeNextCom(cop,x,dt)
        x_cmd=[[c_x,d_c_x] , [c_y,d_c_y]] #command to apply

        if sigmaNoisePosition >0:     
            x[0][0]+=np.random.normal(0,sigmaNoisePosition) #add some disturbance!
            x[1][0]+=np.random.normal(0,sigmaNoisePosition)
        if sigmaNoiseVelocity >0:  
            x[0][1]+=np.random.normal(0,sigmaNoiseVelocity)
            x[1][1]+=np.random.normal(0,sigmaNoiseVelocity)
        if USE_WIIMOTE:
            v[0]=v[0]*0.2 + 0.8*(wm.state['acc'][0]-128)/50.0
            v[1]=v[1]*0.2 + 0.8*(wm.state['acc'][1]-128)/50.0    
        elif USE_GAMEPAD:
            pygame.event.pump()
            v[0]=-my_joystick.get_axis(1)
            v[1]=-my_joystick.get_axis(0)
            if my_joystick.get_button(0) == 1 :
                RUN_FLAG = False
        else : #Stay in the 2mx2m central box
            if c_x>1.0:
                v[0]=-1.0
            if c_x<-1.0:
                v[0]=1.0
            if c_y>1.0:
                v[1]=-1.0
            if c_y<-1.0:
                v[1]=1.0
                
        steps = pg.computeStepsPosition(ev,p0,v,x, LR)
        cop=[steps[0][0],steps[1][0]]

        showStepPreviewInViewer(p.robot,steps)
        currentFoot = p0#[steps[0][0],steps[1][0]]
        nextFoot    = [steps[0][1],steps[1][1]]

        if DISPLAY_PREVIEW:
            [tt, cc_x , cc_y , d_cc_x , d_cc_y] = pg.computePreviewOfCom(steps,ev,x,N=N_COM_TO_DISPLAY) 
            #~ embed()
            showComPreviewInViewer(p.robot,[cc_x,cc_y])
            if ENABLE_LOGING:
                for i in range(len(tt)):
                    tt[i]+=tk
                plt.figure(2)
                plt.plot(tt[:10],cc_x[:10],'-x')
        
        
        [xf,yf,zf,dxf,dyf,dzf] = foot_interpolate(lastFoot[0],lastFoot[1],nextFoot[0],nextFoot[1],ev,durrationOfStep)
        if LR :
            left_foot_xyz    = [ xf, yf, zf]
            left_foot_dxdydz = [dxf,dyf,dzf]
            right_foot_xyz = [p0[0],p0[1],0.0] #current support foot
            right_foot_dxdydz = [0,0,0]
        else :
            right_foot_xyz = [xf,yf,zf]
            right_foot_dxdydz = [dxf,dyf,dzf]
            left_foot_xyz  = [p0[0],p0[1],0.0] #current support foot
            left_foot_dxdydz= [0,0,0]
            
        left_foot=left_foot_xyz[:2]
        right_foot=right_foot_xyz[:2]
        t0=time.time()
        currentCOM,v_currentCOM,err,errDyn = p.controlLfRfCom(left_foot_xyz,
                                        left_foot_dxdydz,
                                        right_foot_xyz,
                                        right_foot_dxdydz,
                                        [x[0][0],x[1][0],h],
                                        [x[0][1],x[1][1],0])
        if ENABLE_LOGING:
            log_t.append(simulationTime)
            log_comx_mesure.append(currentCOM[0,0])
            log_comx_cmd.append (x[0][0])
        #up date the state:
        #option 1: the state is the wanted command
        x = x_cmd
        
        #option 2: the state is the mesure
        #~ x = [[currentCOM[0,0],v_currentCOM[0,0]],[currentCOM[1,0] ,v_currentCOM[1,0]]] # PREVIEW IS CLOSE LOOP
        
        #~ vect_f.append (err[1,0])
        #~ vect_df.append(errDyn[1,0])
        simulationTime+=dt
        #~ print simulationTime
        if (simulationTime>STOP_TIME): 
            RUN_FLAG=False
        ev+=1.0/pps
    #prepare next point
    lastFoot = currentFoot
    p0 = nextFoot 
    x0 = x 
    #~ embed()
    LR = not LR
    ev=0.0
    tk+=durrationOfStep
    
#~ vect_df_findiff = []
#~ tmp=0
#~ for f_tmp in vect_f:
    #~ vect_df_findiff.append((f_tmp-tmp)/dt)
    #~ tmp=f_tmp
    
if USE_WIIMOTE:
    wm.close()
    
if ENABLE_LOGING:
    plt.figure(2)
    plt.hold(True)
    #plt.title("State = command") 
    plt.plot(log_t,log_comx_mesure,'-d',label="COMx mesure")
    plt.plot(log_t,log_comx_cmd,   '-d',label="COMx cmd")
    plt.legend()
    plt.show()


