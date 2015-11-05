#Import modules
import pinocchio as se3
from IPython import embed
import matplotlib.pyplot as plt
import numpy as np
from pinocchio.utils import *
from pinocchio.romeo_wrapper import RomeoWrapper
from pinocchio.reemc_wrapper import ReemcWrapper
import scipy
import time

print ("start")
#load robot ____________________________________________________________
robot = ReemcWrapper("/home/tflayols/devel-src/reemc_wrapper/reemc/reemc.urdf")
robot.initDisplay()
robot.loadDisplayModel("world/pinocchio","pinocchio")
robot.display(robot.q0)
robot.viewer.gui.refresh()
q =np.copy(robot.q0)
v =np.copy(robot.v0)
a =np.copy(robot.v0)

#define const __________________________________________________________
Nstep=4 #number of step in preview
pps=80  #point per step
g=9.81  #(m.s-2) gravity
h=0.80  #(m) Heigth of COM

durrationOfStep=0.8 #(s) time of a step
Dpy=0.20
beta_x=2.0 
beta_y=8.0
dt=durrationOfStep/pps
print( "dt= "+str(dt*1000)+"ms")
v=[1.0,1.0]
#initial feet positions
p0      =[0.0102606,-0.096]
cop=p0
lastFoot=[0.0102606,0.096]

#MPC____________________________________________________________________
# Create matrix for MPC problem
#inputs
initial_com=robot.com(robot.q0)
x0=[[initial_com[0,0],initial_com[1,0]] , [0.0,0.0]]
LR=True     #Left or Right foot flying?
alpha=0.0 #evolution in the phase (0: start, 1: end)
p0_x=p0[0]  #center of foot
p0_y=p0[1]
vx= v[0] # speed command
vy= v[1] # speed command

#others const
gamma=10.0   #Gain : Respect of COP-Center of foot
w2= g/h
w = np.sqrt(w2)

if LR :
    Dpy=-Dpy
def temporal_Fx(T):
    return np.matrix([[np.cosh(w*T)   ,np.sinh(w*T)/w],
                       [np.sinh(w*T)*w ,np.cosh(w*T)  ]])   
def temporal_Fu(T):
    return np.matrix([[ 1-np.cosh(w*T)   ],
                       [-w*np.sinh(w*T)  ]])    

Fx=temporal_Fx(durrationOfStep)                 
Fu=temporal_Fu(durrationOfStep)  
        
tr=durrationOfStep*(1-alpha)
Fx_tr=temporal_Fx(tr)                 
Fu_tr=temporal_Fu(tr)  
        
x0_x=np.matrix([[x0[0][0]],
                [x0[0][1]]])
                        
x0_y=np.matrix([[x0[1][0]],
                [x0[1][1]]])

A_p1 = np.zeros([Nstep,Nstep])
for i in range(Nstep):
    for j in range(0, i+1):
        if (j == 0):
            A_p1[i,j]=(Fx**(i-j)*Fu_tr)[1,0]
        else:
            A_p1[i,j]=(Fx**(i-j)*Fu)[1,0]
A_p2_x = np.zeros([Nstep-1,Nstep])
A_p2_y = np.zeros([Nstep-1,Nstep])
for i in range(Nstep-1):
    A_p2_x[i,i]  = beta_x
    A_p2_x[i,i+1]=-beta_x
    A_p2_y[i,i]  = beta_y
    A_p2_y[i,i+1]=-beta_y        

A_p3=np.zeros([1,Nstep])
A_p3[0,0]=gamma
        
A_p_x=np.vstack([A_p1,A_p2_x,A_p3])
A_p_y=np.vstack([A_p1,A_p2_y,A_p3])

b_p1_x = np.zeros([Nstep,1])
b_p1_y = np.zeros([Nstep,1])
for i in range(Nstep):
    b_p1_x[i]=vx - (Fx_tr *  Fx**(i)*x0_x)[1,0] #- p0_x*(Fx**(i)*temporal_Fu(durrationOfStep*(1-alpha)))[1,0]
    b_p1_y[i]=vy - (Fx_tr *  Fx**(i)*x0_y)[1,0] #- p0_y*(Fx**(i)*temporal_Fu(durrationOfStep*(1-alpha)))[1,0]
b_p2_x = np.zeros([Nstep-1,1])                
b_p2_y = np.zeros([Nstep-1,1])   
for i in range(Nstep-1):
    b_p2_x[i]= beta_x *0
    b_p2_y[i]= beta_y*Dpy*(-1)**i #todo: check!


b_p3_x=np.zeros([1,1])
b_p3_y=np.zeros([1,1])
        
b_p3_x[0,0]=gamma*p0_x
b_p3_y[0,0]=gamma*p0_y

b_p_x=np.vstack([b_p1_x,b_p2_x,b_p3_x])
b_p_y=np.vstack([b_p1_y,b_p2_y,b_p3_y])


#A_MPC = [[ Apx ,  0  ],
#         [  0  , Apy ]]

#b_MPC = [b_p_x 



A_MPC=np.zeros([A_p_x.shape[0]+A_p_y.shape[0],A_p_x.shape[1]+A_p_y.shape[1]])
A_MPC[0:A_p_x.shape[0],0:A_p_x.shape[1]]=A_p_x
A_MPC[A_p_x.shape[0]:A_p_x.shape[0]+A_p_y.shape[0],A_p_x.shape[1]:A_p_x.shape[1]+A_p_y.shape[1]]=A_p_y

b_MPC




#SOLVE QP: ________________________________________________________

#~ p_vect_x=(np.dot(np.linalg.pinv(A_p_x),b_p_x)).T
#~ p_vect_y=(np.dot(np.linalg.pinv(A_p_y),b_p_y)).T 
        
embed()
