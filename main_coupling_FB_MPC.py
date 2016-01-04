from full_body import PinocchioControllerAcceleration

import pinocchio as se3
from IPython import embed
from mpc_foot_position import PgMini
from foot_trajectory_generator import Foot_trajectory_generator
import matplotlib.pyplot as plt
import numpy as np
from qpoases import PyQProblemB as QProblemB
from qpoases import PyQProblem as QProblem
from pinocchio.romeo_wrapper import RomeoWrapper
from pinocchio.reemc_wrapper import ReemcWrapper
from initial_pose_generator import *

import time
print ("start")

N_COM_TO_DISPLAY = 10 #preview: number of point in a phase of COM (no impact on solution, display only)

USE_WIIMOTE=False
USE_GAMEPAD=True
DISPLAY_PREVIEW=False
ENABLE_LOGING=True
ROBOT_MODEL="ROMEO" 
STOP_TIME = 2.0#np.inf

#define const
Nstep=4 #number of step in preview
pps=80  #point per step
g=9.81  #(m.s-2) gravity

if   (ROBOT_MODEL == "ROMEO"):
    h=0.63  #(m) Heigth of COM
elif (ROBOT_MODEL == "REEMC"): 
    h=0.80  #(m) Heigth of COM
fh=0.05 #maximum altitude of foot in flying phases 
ev_foot_const = 0.6# % when the foot target become constant (0.8)
durrationOfStep=0.8#(s) time of a step
Dpy=0.20
beta_x=3.0 #cost on pi-pi+1
beta_y=8.0
gamma=3.0

sigmaNoisePosition=0.00 #optional noise on COM measurement
sigmaNoiseVelocity=0.00
#initialisation of the pg

dt=durrationOfStep/pps
print( "dt= "+str(dt*1000)+"ms")

#load robot model
if   (ROBOT_MODEL == "ROMEO"):
    robot = RomeoWrapper("/local/tflayols/softwares/pinocchio/models/romeo.urdf")
elif (ROBOT_MODEL == "REEMC"): 
    robot = ReemcWrapper("/home/tflayols/devel-src/reemc_wrapper/reemc/reemc.urdf")
robot.initDisplay()
robot.loadDisplayModel("world/pinocchio","pinocchio")

#Initial pose, stand on one feet (com = center of foot)
q_init=compute_initial_pose(robot)
#q_init=robot.q0.copy()

robot.display(q_init)

pg = PgMini(Nstep,g,h,durrationOfStep,Dpy,beta_x,beta_y,gamma)     
p=PinocchioControllerAcceleration(dt,robot,q_init)
ftg=Foot_trajectory_generator(fh,durrationOfStep * (1-ev_foot_const))
v=[1.0,1.0]

#initial feet positions
initial_RF = np.array(  robot.Mrf(q_init).translation  ).flatten().tolist()[:2]
initial_LF = np.array(  robot.Mlf(q_init).translation  ).flatten().tolist()[:2]
p0      =  initial_RF#[0.0102606,-0.096]
cop=p0
lastFoot=  initial_LF#[0.0102606,0.096]
#current foot position, speed and acceleration
[foot_x0  ,foot_y0]  =lastFoot
[foot_dx0 ,foot_dy0] =[0.0,0.0]
[foot_ddx0,foot_ddy0]=[0.0,0.0]

current_flying_foot   = [foot_x0  ,foot_y0  ]
v_current_flying_foot = [foot_dx0 ,foot_dy0 ]
a_current_flying_foot = [foot_ddx0,foot_ddy0]

steps=[[0,0],[0,0]]
#~ steps[0][1] = foot_x0
#~ steps[1][1] = foot_y0
p1=p0
[steps[0][0],steps[1][0]]=cop
[steps[0][1],steps[1][1]]=p1

def cost_on_p1(ev,ev_foot_const):
    if ev > ev_foot_const:
        #~ c=1000
        #c= 1/(1-ev+0.0001) - 1/(1-ev_foot_const+0.0001)
        A=gamma #gain final
        a=A/(ev-ev_foot_const)
        b=A-a
        c=(ev-ev_foot_const)*A/(1-ev_foot_const)
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
prepareCapsForStepPreviewInViewer(robot)
prepareCapsForComPreviewInViewer(robot)
initial_com=robot.com(q_init)
x0=[[initial_com[0,0],0.0] , [initial_com[1,0],0.0]]
x=x0
p1_star=[.0,.0]
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

    log_right_foot_x_mesure=[]
    log_left_foot_x_mesure= []
    
    log_t=[]
    
    log_p0_x=[]
    log_p0_y=[]
    
    log_p1_x=[]
    log_p1_y=[]
    
    log_p1_star_x=[]
    log_p1_star_y=[]
    
    log_cop_x=[]
    log_cop_y=[]
    
RUN_FLAG=True
ev=0.0
tk=0 
it=0
while(RUN_FLAG):
    for ev in np.linspace(0,1-(1.0/pps),pps):
        it+=1
        t=durrationOfStep*ev
        #************************** M P C ******************************
        # Compute matrix for MPC part of the problem
        pg.computeStepsPosition(ev,p0,v,x, LR,p1_star,cost_on_p1(ev,ev_foot_const),False)
        pg.computeNextCom(cop,x,dt)

        #~ showStepPreviewInViewer(robot,steps)
        #~ if DISPLAY_PREVIEW:
            #[tt, cc_x , cc_y , d_cc_x , d_cc_y] = pg.computePreviewOfCom(steps,ev,x,N=N_COM_TO_DISPLAY) 
            #~ embed()
            #~ showComPreviewInViewer(robot,[cc_x,cc_y])
    
        [foot_x1,foot_y1]=[steps[0][1],steps[1][1]] #Goal for the flying foot (needed when t>ev_foot_const)
        [xf,dxf,ddxf  ,  yf,dyf,ddyf  ,  zf,dzf,ddzf , p1_star_x , p1_star_y]= ftg.get_next_foot(  current_flying_foot[0],
                                                                                                 v_current_flying_foot[0],
                                                                                                 a_current_flying_foot[0],
                                                                                                 
                                                                                                   current_flying_foot[1], 
                                                                                                 v_current_flying_foot[1],
                                                                                                 a_current_flying_foot[1], foot_x1, foot_y1, t , durrationOfStep ,  dt)
        p1_star=[p1_star_x,p1_star_y] #Realistic destination (=Goal if we have time... see "ev_foot_const")
        
        #express foot acceleration as linear func of x1,y1
        #~ ddxf=ftg.coeff_acc_x_lin_a * foot_x1 + ftg.coeff_acc_x_lin_b
        #~ ddyf=ftg.coeff_acc_y_lin_a * foot_y1 + ftg.coeff_acc_y_lin_b

        if LR :
            left_foot_xyz    = [ .0, .0, zf]
            left_foot_dxdydz = [.0,.0,dzf]
            left_foot_ddxddyddz=[.0,.0,ddzf]
            right_foot_xyz = [p0[0],p0[1],0.0] #current support foot
            right_foot_dxdydz = [0,0,0]
            right_foot_ddxddyddz=[0,0,0]
        else :
            right_foot_xyz = [.0,.0,zf]
            right_foot_dxdydz = [.0,.0,dzf]
            right_foot_ddxddyddz=[.0,.0,ddzf]
            left_foot_xyz  = [p0[0],p0[1],0.0] #current support foot
            left_foot_dxdydz= [0,0,0]
            left_foot_ddxddyddz=[0,0,0]
        left_foot=left_foot_xyz[:2]
        right_foot=right_foot_xyz[:2]
        t0=time.time()
        
        #******************** F U L L   B O D Y ************************
        qddot = p.controlLfRfCom       (left_foot_xyz,
                                        left_foot_dxdydz,
                                        left_foot_ddxddyddz,
                                        right_foot_xyz,
                                        right_foot_dxdydz,
                                        right_foot_ddxddyddz,
                                        [.0,.0,h],
                                        [.0,.0,0],
                                        [.0,.0,0.0],
                                        LR
                                        )
        #******************** C O U P L I N G  *************************

        #get matrix for the full body
        A_FB = p.A_FB
        b_FB = p.b_FB
        A_MPC = pg.A_MPC
        b_MPC = pg.b_MPC

        #write as one problem
        Zeros1=np.zeros([A_MPC.shape[0], A_FB.shape[1]])
        Zeros2=np.zeros([ A_FB.shape[0], A_MPC.shape[1]])
        A_coupl =  np.vstack([np.hstack([A_MPC  ,Zeros1]),
                                  np.hstack([Zeros2 ,A_FB  ])])
        b_coupl =  np.vstack([b_MPC,
                              b_FB])
        #write as a QP
            
        #Using QPoases:**************************

        #Equality constrains on com (x and y)
        Acom_ = np.hstack([np.zeros([2,A_MPC.shape[1]]),p.Jcom[:2]])
        Acom_[0,0               ]=-pg.coeff_acc_x_lin_a      #p0_x
        Acom_[1,A_MPC.shape[1]/2]=-pg.coeff_acc_y_lin_a      #p0_y
        lb_Acom        = np.array([ pg.coeff_acc_x_lin_b ,pg.coeff_acc_y_lin_b])  
        #Equality constrains on flying foot
        AflyingFoot_   = np.hstack([np.zeros([2,A_MPC.shape[1]]),p.JflyingFoot[:2]])
        AflyingFoot_[0,1                 ]=-ftg.coeff_acc_x_lin_a  #p1_x
        AflyingFoot_[1,1+A_MPC.shape[1]/2]=-ftg.coeff_acc_y_lin_a  #p1_y
        lb_AflyingFoot = np.array([ftg.coeff_acc_x_lin_b,ftg.coeff_acc_y_lin_b])
        A_   = np.vstack([Acom_,AflyingFoot_])
        lb_A = np.hstack([lb_Acom,lb_AflyingFoot])

        ub_A = lb_A

        H=np.array( A_coupl.T*A_coupl).T
        g=np.array(-A_coupl.T*b_coupl).T[0]
           
        lb=-100.0*np.ones(A_coupl.shape[1])
        ub= 100.0*np.ones(A_coupl.shape[1])
        qpb = QProblem(A_coupl.shape[1],A_.shape[0])
        qpb.init(H,g,A_,lb,ub,lb_A,ub_A,np.array([1000]))
        sol=np.zeros(A_coupl.shape[1])
        qpb.getPrimalSolution(sol)
        solution = np.matrix(sol).T
        qddot = solution[-p.robot.nv:] #qddot
        pi_x=solution[     :  Nstep].T.tolist()[0]
        pi_y=solution[Nstep:2*Nstep].T.tolist()[0]
        steps=[pi_x,pi_y]
            
        #~ #wanted acceletarion
        #~ print "*** ACC COM WANTED***"
        #~ print pg.coeff_acc_x_lin_a*steps[0][0]+pg.coeff_acc_x_lin_b
        #~ print pg.coeff_acc_y_lin_a*steps[1][0]+pg.coeff_acc_y_lin_b
        #~ 
        #~ print "*** ACC FF WANTED***"
        #~ print ftg.coeff_acc_x_lin_a * steps[0][1] + ftg.coeff_acc_x_lin_b
        #~ print ftg.coeff_acc_y_lin_a * steps[1][1] + ftg.coeff_acc_y_lin_b
        #~ 
        #~ #test constrains
        #~ print "*** ACC COM REAL***"
        #~ Jcom=robot.Jcom(p.q)
        #~ print Jcom*qddot
            #~ 
        #~ print "*** ACC FF REAL***"
        #~ Jff=p.JflyingFoot
        #~ print Jff*qddot

            
            
        #~ embed()


        #***************A P P L Y I N G   C O N T R O L*****************
        
        p.a = qddot
        p.v += np.matrix(p.a*p.dt)
        p.robot.increment(p.q, np.matrix(p.v*p.dt))

        
        #************* S T A T E   M E A S U R E M E N T ***************
        currentCOM =   p.robot.com(p.q)
        v_currentCOM = p.robot.Jcom(p.q)*p.v

        accLf=p.robot.acceleration(p.q,p.v,p.a,p.robot.lf).linear
        accRf=p.robot.acceleration(p.q,p.v,p.a,p.robot.rf).linear
        
        velLf=p.robot.velocity    (p.q,p.v    ,p.robot.lf).linear
        velRf=p.robot.velocity    (p.q,p.v    ,p.robot.rf).linear
        
        posLf=p.robot.position    (p.q        ,p.robot.lf).translation
        posRf=p.robot.position    (p.q        ,p.robot.rf).translation

        a_current_LF= [accLf[0,0],accLf[1,0]] #acceleration. x,y
        a_current_RF= [accRf[0,0],accRf[1,0]] #acceleration. x,y
            
        v_current_LF= [velLf[0,0],velLf[1,0]] #velocity. x,y
        v_current_RF= [velRf[0,0],velRf[1,0]] #velocity. x,y

        current_LF  = [posLf[0,0],posLf[1,0]] #position. x,y
        current_RF  = [posRf[0,0],posRf[1,0]] #position. x,y

        
        if (not LR):
            current_flying_foot  = current_RF
            v_current_flying_foot  = v_current_RF
            a_current_flying_foot  = a_current_RF
            
            current_support_foot = current_LF
            v_current_support_foot = v_current_LF
            a_current_support_foot = a_current_LF
            
        else:
            current_flying_foot  = current_LF
            v_current_flying_foot  = v_current_LF
            a_current_flying_foot  = a_current_LF
            
            current_support_foot = current_RF
            v_current_support_foot = v_current_RF
            a_current_support_foot = a_current_RF
                
        if (ENABLE_LOGING):
            log_t.append(simulationTime)
            log_right_foot_x.append(right_foot_xyz[0])
            log_left_foot_x.append(  left_foot_xyz[0])
            log_right_foot_x_mesure.append(current_RF[0])
            log_left_foot_x_mesure.append( current_LF[0])     
            log_comx_mesure.append(currentCOM[0,0])
            log_comy_mesure.append(currentCOM[1,0])
            log_vcomx_mesure.append(v_currentCOM[0,0])
            log_vcomy_mesure.append(v_currentCOM[1,0])
            log_p0_x.append(p0[0])
            log_p0_y.append(p0[1])
            log_p1_star_x.append(p1_star[0])
            log_p1_star_y.append(p1_star[1])
            log_p1_x.append(p1[0])
            log_p1_y.append(p1[1])
            log_cop_x.append(cop[0])
            log_cop_y.append(cop[1])


            #~ for i in range(len(tt)):
                #~ tt[i]+=tk
            #~ if (it==50):#(it%1==0):
                #~ plt.figure(1)
                #~ plt.subplot(2,2,1)
                #~ plt.plot(tt,cc_x,'k-x') #actual preview
                #~ plt.subplot(2,2,2)
                #~ plt.plot(tt,cc_y,'k-x') #actual preview           
                #~ plt.subplot(2,2,3)
                #~ plt.plot(tt,d_cc_x,'k-x') #actual preview           
                #~ plt.subplot(2,2,4)
                #~ plt.plot(tt,d_cc_y,'k-x') #actual preview      
#~ 
            #~ log_dd_c_x.append(dd_c_x)
            #~ log_dd_c_y.append(dd_c_y)
#~ 
            #~ log_comx_state.append (    x[0][0])
            #~ log_comx_cmd.append   (x_cmd[0][0])
            #~ log_comy_state.append (    x[1][0])
            #~ log_comy_cmd.append   (x_cmd[1][0])
            #~ log_vcomx_state.append(    x[0][1])
            #~ log_vcomx_cmd.append  (x_cmd[0][1])
            #~ log_vcomy_state.append(    x[1][1])
            #~ log_vcomy_cmd.append  (x_cmd[1][1])       
            





        x = [[currentCOM[0,0],v_currentCOM[0,0]],[currentCOM[1,0] ,v_currentCOM[1,0]]] # PREVIEW IS CLOSE LOOP
        
        #add some disturbance on COM measurements
        if sigmaNoisePosition >0:     
            x[0][0]+=np.random.normal(0,sigmaNoisePosition) 
            x[1][0]+=np.random.normal(0,sigmaNoisePosition)
        if sigmaNoiseVelocity >0:  
            x[0][1]+=np.random.normal(0,sigmaNoiseVelocity)
            x[1][1]+=np.random.normal(0,sigmaNoiseVelocity)
            
        #~ x[0][0]+=disturb_cx
        #~ x[1][0]+=disturb_cy
        #~ x[0][1]+=disturb_dcx
        #~ x[1][1]+=disturb_dcy
        #RAZ eventual disturb
        disturb_cx=0.0
        disturb_cy=0.0
        disturb_dcx=0.0
        disturb_dcy=0.0

        simulationTime+=dt
        
        #~ print simulationTime
        
        
        #******************** READ COMMAND *****************************
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
            if currentCOM[0]>1.0:
                v[0]=-1.0
            if currentCOM[0]<-1.0:
                v[0]=1.0
            if currentCOM[1]>1.0:
                v[1]=-1.0
            if currentCOM[1]<-1.0:
                v[1]=1.0

        #******************** UPDATE DISPLAY ****************************
        if (it%2==0):
            robot.display(p.q)
            robot.viewer.gui.refresh()
            showStepPreviewInViewer(robot,steps)
            #~ showComPreviewInViewer(robot,[cc_x,cc_y])
        
        if (simulationTime>STOP_TIME): 
            RUN_FLAG=False
        #~ ev+=1.0/pps
    #prepare next point
    p0=current_flying_foot 
    LR = not LR
    if (not LR): #Duplicated code...
        current_flying_foot  = current_RF
        v_current_flying_foot  = v_current_RF
        a_current_flying_foot  = a_current_RF
            
        current_support_foot = current_LF
        v_current_support_foot = v_current_LF
        a_current_support_foot = a_current_LF

    else:
        current_flying_foot  = current_LF
        v_current_flying_foot  = v_current_LF
        a_current_flying_foot  = a_current_LF
            
        current_support_foot = current_RF
        v_current_support_foot = v_current_RF
        a_current_support_foot = a_current_RF
    ev=0.0
    tk+=durrationOfStep
    
    
    
if USE_WIIMOTE:
    wm.close()
    
if ENABLE_LOGING:
    embed()
    #Plot COM and dCOM
    plt.figure(1)
    plt.hold(True)
    log_tp1=log_t[1:] #log_tp1 is the timing vector from 1*dt to the end
    log_tp1.append(simulationTime)
    plt.subplot(2,2,1)
    plt.plot(log_tp1,log_comx_mesure,   '-d',label="COMx measure")
    #~ plt.plot(log_tp1,log_comx_cmd,      '-d',label="COMx cmd")
    #~ plt.plot(log_t,log_comx_state,     '-.d',label="COMx state")
    plt.legend()
    plt.subplot(2,2,2)
    plt.plot(log_tp1,log_comy_mesure,   '-d',label="COMy measure")
    #~ plt.plot(log_tp1,log_comy_cmd,      '-d',label="COMy cmd")
    #~ plt.plot(log_t,log_comy_state,     '-.d',label="COMy state")
    plt.legend()
    plt.subplot(2,2,3)
    plt.plot(log_tp1,log_vcomx_mesure,   '-d',label="VCOMx measure")
    #~ plt.plot(log_tp1,log_vcomx_cmd,      '-d',label="VCOMx cmd")
    #~ plt.plot(log_t,log_vcomx_state,     '-.d',label="VCOMx state")
    plt.legend()
    plt.subplot(2,2,4)
    plt.plot(log_tp1,log_vcomy_mesure,   '-d',label="VCOMy measure")
    #~ plt.plot(log_tp1,log_vcomy_cmd,      '-d',label="VCOMy cmd")
    #~ plt.plot(log_t,log_vcomy_state,      '-.d',label="VCOMy state")
    plt.legend()

    #plot feet trajectories
    plt.figure()
    #~ plt.plot(log_t,log_right_foot_x,label="Right foot x")
    #~ plt.plot(log_t, log_left_foot_x,label="Left foot x")
    
    plt.plot(log_t,log_right_foot_x_mesure,label="Right foot x measure")
    plt.plot(log_t, log_left_foot_x_mesure,label="Left foot x measure")
    
    plt.legend()
    plt.figure()
    plt.plot(log_t,log_p0_x,label="p0 x")
    plt.plot(log_t,log_p1_star_x,label="p1* x")
    plt.plot(log_t,log_p1_x,label="p1 x")
    
    plt.plot(log_t,log_cop_x,label="cop x")
    #plt.plot(log_t, log_left_foot_x,label="Left foot x")
    plt.legend()
    
    plt.show()

