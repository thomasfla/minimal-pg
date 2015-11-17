from pinocchio_controller_acceleration_noPD_acc import PinocchioControllerAcceleration
import pinocchio as se3
from IPython import embed
from minimal_pg_relaxed_zmp import PgMini
from foot_trajectory_generator import Foot_trajectory_generator
import matplotlib.pyplot as plt
import numpy as np
import time
print ("start")

#define const
Nstep=4 #number of step in preview
pps=80  #point per step
g=9.81  #(m.s-2) gravity
#~ h=0.63  #(m) Heigth of COM
h=0.80  #(m) Heigth of COM
fh=0.1 #maximum altitude of foot in flying phases 
ev_foot_const = 0.95# % when the foot target become constant (0.8)
durrationOfStep=0.8#(s) time of a step
Dpy=0.20

beta_x=3.0 
beta_y=8.0

N_COM_TO_DISPLAY = 10 #preview: number of point in a phase of COM (no impact on solution, display only)

USE_WIIMOTE=False
USE_GAMEPAD=False
DISPLAY_PREVIEW=True
ENABLE_LOGING=True
STOP_TIME = np.inf

sigmaNoisePosition=0.00 #optional noise on COM measurement
sigmaNoiseVelocity=0.00
#initialisation of the pg

dt=durrationOfStep/pps
print( "dt= "+str(dt*1000)+"ms")
pg = PgMini(Nstep,g,h,durrationOfStep,Dpy,beta_x,beta_y)     
p=PinocchioControllerAcceleration(dt)
ftg=Foot_trajectory_generator(fh,durrationOfStep * (1-ev_foot_const))
v=[1.0,1.0]
#initial feet positions
p0      =[0.0102606,-0.096]
cop=p0
lastFoot=[0.0102606,0.096]

#current foot position, speed and acceleration
[foot_x0  ,foot_y0]  =lastFoot
[foot_dx0 ,foot_dy0] =[0.0,0.0]
[foot_ddx0,foot_ddy0]=[0.0,0.0]

def cost_on_p1(ev,ev_foot_const):
    if ev > ev_foot_const:
        c=1000
        #~ c= 1/(1-ev+0.0001) - 1/(1-ev_foot_const+0.0001)
    else:
        c=0.0
    return c

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
p1=[.0,.0]
comx=[]
comy=[]

vect_f=[]
vect_df=[]
LR=True
#plt.ion()
t0=time.time()
simulationTime=0.0

log_f_poly=[]

disturb_cx=.0
disturb_cy=.0
disturb_dcx=.0
disturb_dcy=.0
if ENABLE_LOGING:
    log_comx_mesure=[]
    log_comx_cmd=[]
    log_comx_state=[]
    
    log_comy_mesure=[]
    log_comy_cmd=[]
    log_comy_state=[]
    
    log_vcomx_mesure=[]
    log_vcomx_cmd=[]
    log_vcomx_state=[]
    
    log_vcomy_mesure=[]
    log_vcomy_cmd=[]
    log_vcomy_state=[]
    
    log_dd_c_x=[]
    log_dd_c_y=[]
    
    log_right_foot_x=[]
    log_left_foot_x =[]
    
    log_t=[]
    

RUN_FLAG=True
ev=0.0
tk=0 
while(RUN_FLAG):
    while(ev<1.0 and RUN_FLAG):
        #~ time.sleep(.2)
        t=durrationOfStep*ev
        #solve MPC for current state x
        
        
        '''extract 1st command to apply, cop position and preview position of com'''
        steps = pg.computeStepsPosition(ev,p0,v,x, LR,p1,cost_on_p1(ev,ev_foot_const),True)
        cop=[steps[0][0],steps[1][0]]
        [c_x , c_y , d_c_x , d_c_y]     = pg.computeNextCom(cop,x,dt)

        w2= pg.g/pg.h
        dd_c_x = w2*( c_x - cop[0] )
        dd_c_y = w2*( c_y - cop[1] )

        x_cmd=[[c_x,d_c_x] , [c_y,d_c_y]] #command to apply
        [tt, cc_x , cc_y , d_cc_x , d_cc_y] = pg.computePreviewOfCom(steps,ev,x,N=N_COM_TO_DISPLAY) 
        if ENABLE_LOGING:
            for i in range(len(tt)):
                tt[i]+=tk
            #~ plt.subplot(2,2,1)
            #~ plt.plot(tt,cc_x,'.') #actual preview
            #~ plt.subplot(2,2,2)
            #~ plt.plot(tt,cc_y,'.') #actual preview           
            #~ plt.subplot(2,2,3)
            #~ plt.plot(tt,d_cc_x,'.') #actual preview           
            #~ plt.subplot(2,2,4)
            #~ plt.plot(tt,d_cc_y,'.') #actual preview      
            
            log_dd_c_x.append(dd_c_x)
            log_dd_c_y.append(dd_c_y)
                    
            log_comx_state.append(    x[0][0])
            log_comx_cmd.append  (x_cmd[0][0])
            log_comy_state.append(    x[1][0])
            log_comy_cmd.append  (x_cmd[1][0])
            
            log_vcomx_state.append(    x[0][1])
            log_vcomx_cmd.append  (x_cmd[0][1])
            log_vcomy_state.append(    x[1][1])
            log_vcomy_cmd.append  (x_cmd[1][1])       
            log_t.append(simulationTime)

        if USE_WIIMOTE:
            v[0]=v[0]*0.2 + 0.8*(wm.state['acc'][0]-128)/50.0
            v[1]=v[1]*0.2 + 0.8*(wm.state['acc'][1]-128)/50.0    
        elif USE_GAMEPAD:
            pygame.event.pump()
            v[0]=-my_joystick.get_axis(1)
            v[1]=-my_joystick.get_axis(0)
            if my_joystick.get_button(0) == 1 :
                RUN_FLAG = False
            if my_joystick.get_button(4) == 1 :
                print "perturbation on : Cx - Cy !"  
                disturb_cx=-my_joystick.get_axis(4)/10.0
                disturb_cy=-my_joystick.get_axis(3)/10.0
            if my_joystick.get_button(5) == 1 :   
                print "perturbation on : dCx - dCy !" 
                disturb_dcx=-my_joystick.get_axis(4)/10.0
                disturb_dcy=-my_joystick.get_axis(3)/10.0
        else : #Stay in the 2mx2m central box
            if c_x>1.0:
                v[0]=-1.0
            if c_x<-1.0:
                v[0]=1.0
            if c_y>1.0:
                v[1]=-1.0
            if c_y<-1.0:
                v[1]=1.0

        showStepPreviewInViewer(p.robot,steps)
        if DISPLAY_PREVIEW:
            #[tt, cc_x , cc_y , d_cc_x , d_cc_y] = pg.computePreviewOfCom(steps,ev,x,N=N_COM_TO_DISPLAY) 
            #~ embed()
            showComPreviewInViewer(p.robot,[cc_x,cc_y])
    
        [foot_x1,foot_y1]=[steps[0][1],steps[1][1]]
        [foot_x0,foot_dx0,foot_ddx0  ,  foot_y0,foot_dy0,foot_ddy0  ,  foot_z0,foot_dz0,foot_ddz0 , p1_x , p1_y]= ftg.get_next_foot(foot_x0, foot_dx0, foot_ddx0, foot_y0, foot_dy0, foot_ddy0, foot_x1, foot_y1, t , durrationOfStep ,  dt)
        p1=[p1_x,p1_y]
        [xf ,yf ,zf ]=[foot_x0,foot_y0,foot_z0]
        [dxf,dyf,dzf]=[foot_dx0,foot_dy0,foot_dz0]

        


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
                                        [x_cmd[0][0],x_cmd[1][0],h],
                                        [x_cmd[0][1],x_cmd[1][1],0],
                                        [dd_c_x,dd_c_y,0.0]
                                        )
        if (ENABLE_LOGING):
            log_right_foot_x.append(right_foot_xyz[0])
            log_left_foot_x.append(  left_foot_xyz[0])
            log_comx_mesure.append(currentCOM[0,0])
            log_comy_mesure.append(currentCOM[1,0])
            log_vcomx_mesure.append(v_currentCOM[0,0])
            log_vcomy_mesure.append(v_currentCOM[1,0])
        #update the state:
        #option 1: the state is the wanted command___________
        #~ x = x_cmd
        #option 2: the state is the measure__________________
        x = [[currentCOM[0,0],v_currentCOM[0,0]],[currentCOM[1,0] ,v_currentCOM[1,0]]] # PREVIEW IS CLOSE LOOP
        
        #add some disturbance on COM measurements
        if sigmaNoisePosition >0:     
            x[0][0]+=np.random.normal(0,sigmaNoisePosition) 
            x[1][0]+=np.random.normal(0,sigmaNoisePosition)
        if sigmaNoiseVelocity >0:  
            x[0][1]+=np.random.normal(0,sigmaNoiseVelocity)
            x[1][1]+=np.random.normal(0,sigmaNoiseVelocity)
            
        x[0][0]+=disturb_cx
        x[1][0]+=disturb_cy
        x[0][1]+=disturb_dcx
        x[1][1]+=disturb_dcy
        #RAZ eventual disturb
        disturb_cx=0.0
        disturb_cy=0.0
        disturb_dcx=0.0
        disturb_dcy=0.0
        
        #~ vect_f.append (err[1,0])
        #~ vect_df.append(errDyn[1,0])
        simulationTime+=dt
        #~ print simulationTime
        if (simulationTime>STOP_TIME): 
            RUN_FLAG=False
        ev+=1.0/pps
    #prepare next point
    #current foot position, speed and acceleration
    [foot_x0  ,foot_y0]  =p0
    [foot_dx0 ,foot_dy0] =[0.0,0.0]
    [foot_ddx0,foot_ddy0]=[0.0,0.0]

    p0=[xf, yf]
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
    #Plot COM and dCOM
    plt.figure()
    plt.hold(True)
    log_tp1=log_t[1:] #log_tp1 is the timing vector from 1*dt to the end
    log_tp1.append(simulationTime)
    plt.subplot(2,2,1)
    plt.plot(log_tp1,log_comx_mesure,   '-d',label="COMx mesure")
    plt.plot(log_tp1,log_comx_cmd,      '-d',label="COMx cmd")
    plt.plot(log_t,log_comx_state,     '-.d',label="COMx state")
    plt.legend()
    plt.subplot(2,2,2)
    plt.plot(log_tp1,log_comy_mesure,   '-d',label="COMy mesure")
    plt.plot(log_tp1,log_comy_cmd,      '-d',label="COMy cmd")
    plt.plot(log_t,log_comy_state,     '-.d',label="COMy state")
    plt.legend()
    plt.subplot(2,2,3)
    plt.plot(log_tp1,log_vcomx_mesure,   '-d',label="VCOMx mesure")
    plt.plot(log_tp1,log_vcomx_cmd,      '-d',label="VCOMx cmd")
    plt.plot(log_t,log_vcomx_state,     '-.d',label="VCOMx state")
    plt.legend()
    plt.subplot(2,2,4)
    plt.plot(log_tp1,log_vcomy_mesure,   '-d',label="VCOMy mesure")
    plt.plot(log_tp1,log_vcomy_cmd,      '-d',label="VCOMy cmd")
    plt.plot(log_t,log_vcomy_state,      '-.d',label="VCOMy state")
    plt.legend()

    #plot feet trajectories
    plt.figure()
    plt.plot(log_t,log_right_foot_x,label="Right foot x")
    plt.plot(log_t, log_left_foot_x,label="Left foot x")
    plt.legend()

    plt.show()

