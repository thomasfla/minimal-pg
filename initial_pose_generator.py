import pinocchio as se3
import numpy as np
from pinocchio.utils import *
from pinocchio.romeo_wrapper import RomeoWrapper
#~ from pinocchio.reemc_wrapper import ReemcWrapper
from IPython import embed
'''Provide a position (q) of the robot joints close to q0 and with CoM on the foot center'''


TESTING=False

def errorInSE3( M,Mdes):
    '''
    Compute a 6-dim error vector (6x1 np.maptrix) caracterizing the difference
    between M and Mdes, both element of SE3.
    '''
    error = se3.log(Mdes.inverse()*M)
    return error.vector
    
    
def robotint(q,dq):
    M = se3.SE3(se3.Quaternion(q[6,0],q[3,0],q[4,0],q[5,0]).matrix(),q[:3])
    dM = se3.exp(dq[:6])
    M = M*dM
    q[:3] = M.translation
    q[3:7] = se3.Quaternion(M.rotation).coeffs()
    q[7:] += dq[6:]

def compute_initial_pose(robot):
    a=0.5
    eps = 1e-9
    
    initial_RF = np.array(  robot.Mrf(robot.q0).translation  ).flatten().tolist()[:2]
    initial_LF = np.array(  robot.Mlf(robot.q0).translation  ).flatten().tolist()[:2]
    
    com_des = robot.com(robot.q0)
    com_des[0]=initial_RF[0]
    com_des[1]=initial_RF[1]

    Rf_des=robot.Mrf(robot.q0)
    Lf_des=robot.Mlf(robot.q0)
    for i in range(1000):
        q=robot.q0.copy()
        
        #COM ***********************
        err_com=robot.com(q)-com_des
        J_com =robot.Jcom(q)
        
        #LF ************************
        err_lf=errorInSE3(robot.Mlf(q), Lf_des)
        J_lf = robot.Jlf(q)
        
        #RF ************************
        err_rf=errorInSE3(robot.Mrf(q), Rf_des)
        J_rf = robot.Jrf(q)
        
        #POSTURE *******************
        err_post = (q-robot.q0)[7:]
        J_post = np.hstack( [ zero([robot.nv-6,6]), eye(robot.nv-6) ] )
        
        J=np.vstack(  [J_com  ,J_lf  ,J_rf,J_post     * eps]  )
        err=np.vstack([err_com,err_lf,err_rf,err_post * eps])

        dq=np.linalg.pinv(J)*(-a*err)
        robotint(q,dq)

        #print err
    return q
if (TESTING) :
    #robot = RomeoWrapper("/local/tflayols/softwares/pinocchio/models/romeo.urdf")
    robot = ReemcWrapper("/home/tflayols/devel-src/reemc_wrapper/reemc/reemc.urdf")
    robot.initDisplay()
    robot.loadDisplayModel("world/pinocchio","pinocchio")
    robot.display(robot.q0)
    robot.viewer.gui.refresh()




    q=compute_initial_pose(robot)
    robot.display(q)
    robot.viewer.gui.refresh()
    embed()
