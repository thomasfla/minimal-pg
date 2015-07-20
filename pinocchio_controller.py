import pinocchio as se3
import numpy as np
from pinocchio.utils import *
from pinocchio.romeo_wrapper import RomeoWrapper
import scipy
class PinocchioController(object):
    def __init__(self):
        self.robot = RomeoWrapper("/local/tflayols/softwares/pinocchio/models/romeo.urdf")
        self.robot.initDisplay()
        self.robot.loadDisplayModel("world/pinocchio","pinocchio")
        self.robot.display(self.robot.q0)
        self.robot.viewer.gui.refresh()
        self.q=np.copy(self.robot.q0)
    def controlLfRfCom(self,Lf=[0.0,0.0,0.0],Rf=[0.0,0.0,0.0],Com=[0,0,0.63],K=1.0):
        def robotint(q,dq):
            M = se3.SE3( se3.Quaternion(q[6,0],q[3,0],q[4,0],q[5,0]).matrix(),q[:3])
            dM = se3.exp(dq[:6])
            M = M*dM
            q[:3] = M.translation
            q[3:7] = se3.Quaternion(M.rotation).coeffs()
            q[7:] += dq[6:]
        def errorInSE3( M,Mdes):
            '''
            Compute a 6-dim error vector (6x1 np.maptrix) caracterizing the difference
            between M and Mdes, both element of SE3.
            '''
            error = se3.log(M.inverse()*Mdes)
            return error.vector()
        def null(A, eps=1e-12):
            '''Compute a base of the null space of A.'''
            u, s, vh = np.linalg.svd(A)
            padding = max(0,np.shape(A)[1]-np.shape(s)[0])
            null_mask = np.concatenate(((s <= eps), np.ones((padding,),dtype=bool)),axis=0)
            null_space = scipy.compress(null_mask, vh, axis=0)
            return scipy.transpose(null_space)
        XYZ_LF=np.array(Lf)+np.array([.0,.0,0.07])
        RPY_LF=np.matrix([[.0],[.0],[.0]])
        SE3_LF=se3.SE3(se3.utils.rpyToMatrix(RPY_LF),XYZ_LF)
        
        XYZ_RF=np.array(Rf)+np.array([.0,.0,0.07])
        RPY_RF=np.matrix([[.0],[.0],[.0]])
        SE3_RF=se3.SE3(se3.utils.rpyToMatrix(RPY_RF),XYZ_RF)


        #_RF________________________________________________________________
        Jlf=self.robot.Jlf(self.q).copy()
        Jlf[:3] = self.robot.Mlf(self.q).rotation * Jlf[:3,:]#Orient in the world base
        errRf = errorInSE3(SE3_RF,self.robot.Mrf(self.q))
        #_LF________________________________________________________________    
        Jrf=self.robot.Jrf(self.q).copy()
        Jrf[:3] = self.robot.Mrf(self.q).rotation * Jrf[:3,:]#Orient in the world base
        errLf = errorInSE3(SE3_LF,self.robot.Mlf(self.q))
        #_COM_______________________________________________________________
        Jcom=self.robot.Jcom(self.q)[:3]
        errCOM = self.robot.com(self.q)[:3]-np.matrix(Com).T
        print self.robot.com(self.q)[:3]
        print np.array(Com).T
        #_TASK1 STACK_______________________________________________________
        print errLf.shape
        print errRf.shape
        print errCOM.shape
        err1 = np.vstack([errLf,errRf,errCOM])

        J1 = np.vstack([Jlf,Jrf,Jcom])
        #_Posture___________________________________________________________
        Jpost = np.hstack( [ zero([self.robot.nv-6,6]), eye(self.robot.nv-6) ] )
        errpost =  -1 * (self.q-self.robot.q0)[7:]
        
        #_TASK2 STACK_______________________________________________________
        err2 = errpost
        J2 = Jpost

        #Hierarchical solve_________________________________________________
        qdot = npl.pinv(J1)*-K * err1
        Z = null(J1)
        qdot += Z*npl.pinv(J2*Z)*(K*err2 - J2*qdot)
        #__Integration______________________________________________________
        robotint(self.q,qdot)
        self.robot.display(self.q)
        self.robot.viewer.gui.refresh()
        
        return self.robot.com(self.q)
        
p=PinocchioController()
p.controlLfRfCom()

