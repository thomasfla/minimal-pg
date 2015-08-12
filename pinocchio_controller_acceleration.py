import pinocchio as se3
import numpy as np
from pinocchio.utils import *
from pinocchio.romeo_wrapper import RomeoWrapper
import scipy
from IPython import embed
class PinocchioControllerAcceleration(object):
    def __init__(self,dt):
        
        self.dt=dt
        self.robot = RomeoWrapper("/local/tflayols/softwares/pinocchio/models/romeo.urdf")
        self.robot.initDisplay()
        self.robot.loadDisplayModel("world/pinocchio","pinocchio")
        self.robot.display(self.robot.q0)
        self.robot.viewer.gui.refresh()
        self.q =np.copy(self.robot.q0)
        self.v =np.copy(self.robot.v0)
        self.dq=np.matrix(np.zeros([self.robot.nv,1]))
    def controlLfRfCom(self,Lf=[.0,.0,.0],dLf=[.0,.0,.0],Rf=[.0,.0,.0],dRf=[.0,.0,.0],Com=[0,0,0.63],dCom=[.0,.0,.0],K=1.0):
        def robotint(q,dq):
            M = se3.SE3( se3.Quaternion(q[6,0],q[3,0],q[4,0],q[5,0]).matrix(),q[:3])
            dM = se3.exp(dq[:6])
            M = M*dM
            q[:3] = M.translation
            q[3:7] = se3.Quaternion(M.rotation).coeffs()
            q[7:] += dq[6:]
        def robotdoubleint(q,dq,ddq,dt):
            
            dq += dt*ddq
            robotint(q,dq)
        def errorInSE3( M,Mdes):
            '''
            Compute a 6-dim error vector (6x1 np.maptrix) caracterizing the difference
            between M and Mdes, both element of SE3.
            '''
            error = se3.log(M.inverse()*Mdes)
            return error.vector()
            
        def errorInSE3dyn(M,Mdes,v_frame,v_des):
            gMl = se3.SE3.Identity()
            gMl.rotation = M.rotation
            # Compute error
            error = errorInSE3(M, Mdes);
            v_error = v_frame - gMl.actInv(v_des)
            return error,v_error.vector()
            
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
        Jrf=self.robot.Jrf(self.q).copy()
        Jrf[:3] = self.robot.Mrf(self.q).rotation * Jrf[:3,:]#Orient in the world base
        #errRf = errorInSE3   (SE3_RF,self.robot.Mrf(self.q))
        v_frame = self.robot.velocity(self.q,self.v,self.robot.rf)
        v_ref= se3.se3.Motion(np.matrix([dRf[0],dRf[1],dRf[2],.0,.0,.0]).T)
        errRf,v_errRf = errorInSE3dyn(SE3_RF,self.robot.Mrf(self.q),v_frame,v_ref)

        #_LF________________________________________________________________    
        Jlf=self.robot.Jlf(self.q).copy()
        Jlf[:3] = self.robot.Mlf(self.q).rotation * Jlf[:3,:]#Orient in the world base
        #errLf = errorInSE3(SE3_LF,self.robot.Mlf(self.q))
        v_frame = self.robot.velocity(self.q,self.v,self.robot.lf)
        v_ref= se3.se3.Motion(np.matrix([dLf[0],dLf[1],dLf[2],.0,.0,.0]).T)
        errLf,v_errLf = errorInSE3dyn(SE3_LF,self.robot.Mlf(self.q),v_frame,v_ref)
        
        #_COM_______________________________________________________________
        errCOM = self.robot.com(self.q)[:3]-(np.matrix(Com).T)[:3]
        Jcom=self.robot.Jcom(self.q)[:3]
        v_com = Jcom*self.v
        v_errCOM= v_com - (np.matrix(dCom).T)[:3]
        #_Trunk_____________________________________________________________
        idx_Trunk = self.robot.index('root')

        #embed()
        MTrunk0=self.robot.position(self.robot.q0,idx_Trunk)
        MTrunk=self.robot.position(self.q,idx_Trunk)
        errTrunk=errorInSE3(MTrunk0,MTrunk)[3:6]
        JTrunk=self.robot.jacobian(self.q,idx_Trunk)[3:6]
        v_frame = self.robot.velocity(self.q,self.v,idx_Trunk)
        v_ref= se3.se3.Motion(np.matrix([.0,.0,.0,.0,.0,.0]).T)
        errTrunk,v_errTrunk = errorInSE3dyn(SE3_LF,MTrunk,v_frame,v_ref)
        errTrunk=errTrunk[3:6]
        v_errTrunk=v_errTrunk[3:6]

    #_TASK1 STACK_______________________________________________________
        err1 = np.vstack([errLf,errRf,errCOM,errTrunk])
        v_err1 = np.vstack([v_errLf,v_errRf,v_errCOM,v_errTrunk])
        
        J1 = np.vstack([Jlf,Jrf,Jcom,JTrunk])
        #~ #_Posture___________________________________________________________
        Jpost = np.hstack( [ zero([self.robot.nv-6,6]), eye(self.robot.nv-6) ] )
        errPost = (self.q-self.robot.q0)[7:]
        v_errPost = (self.v-self.robot.v0)[6:]
        
        #~ errpost =  -1 * (self.q-self.robot.q0)[7:]
        #~ embed()
        
    #_TASK2 STACK_______________________________________________________
        err2 = errPost
        v_err2 = v_errPost
        J2 = Jpost
        #Hierarchical solve_________________________________________________
        #qdot = npl.pinv(J1)*-K * err1

        #~ self.q[7:]
        #~ self.dq[6:]

        
        
        
        Kp = K
        Kd = 2*np.sqrt(Kp)

        qddot = npl.pinv(J1)*(-Kp * err1 -Kd * v_err1)
        
        Z = null(J1)
        qddot += Z*npl.pinv(J2*Z)*(-(Kp * err2 + Kd * v_err2) - J2*qddot)
        #__Integration______________________________________________________
        #self.v=qdot/self.dt
        #robotint(self.q,qdot)

        robotdoubleint(self.q,self.dq,qddot,self.dt)
        self.v=self.dq/self.dt
        self.robot.display(self.q)
        self.robot.viewer.gui.refresh()
        return self.robot.com(self.q) ,errCOM, v_errCOM
        

