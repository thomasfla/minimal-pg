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
        
    def controlLfRfCom(self,Lf=[.0,.0,.0],dLf=[.0,.0,.0],Rf=[.0,.0,.0],dRf=[.0,.0,.0],Com=[0,0,0.63],dCom=[.0,.0,.0]):
        def robotint(q,dq):
            M = se3.SE3(se3.Quaternion(q[6,0],q[3,0],q[4,0],q[5,0]).matrix(),q[:3])
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
            error = se3.log(Mdes.inverse()*M)
            return error.vector()
            
        def errorInSE3dyn(M,Mdes,v_frame,v_des):
            gMl = se3.SE3.Identity()
            gMl.rotation = M.rotation
            # Compute error
            error = errorInSE3(M, Mdes);
            v_error = v_frame - gMl.actInv(v_des)

            #~ a_corriolis = self.robot.acceleration(q,v,0*v,self._link_id, update_geometry)
            #~ a_corriolis.linear += np.cross(v_frame.angular.T, v_frame.linear.T).T
            #~ a_tot = a_ref - gMl.actInv(a_corriolis)
            return error,v_error.vector()
            
        def errorLinkInSE3dyn(linkId,Mdes,v_des,q,v):
            # Get the current configuration of the link
            M = self.robot.position(q, linkId)
            gMl = se3.SE3.Identity()
            gMl.rotation = M.rotation
            v_frame = self.robot.velocity(q,v,linkId)
            # Compute error
            error = errorInSE3(M, Mdes);
            v_error = v_frame - gMl.actInv(v_des)

            a_corriolis = self.robot.acceleration(q,v,0*v,linkId)
            #~ a_corriolis.linear += np.cross(v_frame.angular.T, v_frame.linear.T).T

            #~ a_tot = gMl.actInv(a_corriolis) #a_ref - gMl.actInv(a_corriolis)
            a_tot = a_corriolis
            #~ dJdq = a_tot.vector() *self.dt
            dJdq = a_corriolis.vector() 
            return error,v_error.vector() ,dJdq

        def null(A, eps=1e-6):#-12
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
        v_ref= se3.se3.Motion(np.matrix([dRf[0],dRf[1],dRf[2],.0,.0,.0]).T)
        errRf,v_errRf,dJdqRf = errorLinkInSE3dyn(self.robot.rf,SE3_RF,v_ref,self.q,self.v)
        #_LF________________________________________________________________    
        Jlf=self.robot.Jlf(self.q).copy()
        Jlf[:3] = self.robot.Mlf(self.q).rotation * Jlf[:3,:]#Orient in the world base
        v_ref= se3.se3.Motion(np.matrix([dLf[0],dLf[1],dLf[2],.0,.0,.0]).T)
        errLf,v_errLf,dJdqLf = errorLinkInSE3dyn(self.robot.lf,SE3_LF,v_ref,self.q,self.v)
        
        #_COM_______________________________________________________________
        Jcom=self.robot.Jcom(self.q)
        p_com, v_com, a_com = self.robot.com(self.q,self.v,0*self.v)
        errCOM = self.robot.com(self.q)-(np.matrix(Com).T)
        #~ v_com = Jcom*self.v
        v_errCOM= v_com - (np.matrix(dCom).T)
        dJdqCOM=a_com
        #_Trunk_____________________________________________________________
        idx_Trunk = self.robot.index('root')

        MTrunk0=self.robot.position(self.robot.q0,idx_Trunk)
        MTrunk=self.robot.position(self.q,idx_Trunk)
        #errTrunk=errorInSE3(MTrunk0,MTrunk)[3:6]
        JTrunk=self.robot.jacobian(self.q,idx_Trunk)[3:6]
        #v_frame = self.robot.velocity(self.q,self.v,idx_Trunk)
        #v_ref= se3.se3.Motion(np.matrix([.0,.0,.0,.0,.0,.0]).T)
        #errTrunk,v_errTrunk = errorInSE3dyn(MTrunk,MTrunk0,v_frame,v_ref)
        errTrunk,v_errTrunk,dJdqTrunk = errorLinkInSE3dyn(idx_Trunk,MTrunk0,v_ref,self.q,self.v)
        errTrunk=errTrunk[3:6]
        v_errTrunk=v_errTrunk[3:6]
        dJdqTrunk=dJdqTrunk[3:6]


    #_TASK1 STACK_______________________________________________________
        K=1000.0
        Kp_foot=K
        Kp_com=K
        Kp_Trunk=K
        Kp_post=K
        
        Kd_foot= 2*np.sqrt(Kp_foot)
        Kd_com=  2*np.sqrt(Kp_com )
        Kd_Trunk=2*np.sqrt(Kp_Trunk) 
        Kd_post= 2*np.sqrt(Kp_post ) 
        
    
        err1 =   np.vstack([Kp_foot*  errLf, Kp_foot*  errRf, Kp_com*  errCOM, Kp_Trunk*  errTrunk])
        v_err1 = np.vstack([Kd_foot*v_errLf, Kd_foot*v_errRf, Kd_com*v_errCOM, Kd_Trunk*v_errTrunk])
        dJdq1=   np.vstack([         dJdqLf,          dJdqRf,         dJdqCOM,           dJdqTrunk])

        J1 = np.vstack([Jlf,Jrf,Jcom,JTrunk])
        #~ #_Posture___________________________________________________________
        Jpost = np.hstack( [ zero([self.robot.nv-6,6]), eye(self.robot.nv-6) ] )
        errPost =   Kp_post*(self.q-self.robot.q0)[7:]
        v_errPost = Kd_post*(self.v-self.robot.v0)[6:]
        
        errpost =  -1 * (self.q-self.robot.q0)[7:]

    #_TASK2 STACK_______________________________________________________
        err2 = errPost
        v_err2 = v_errPost
        J2 = Jpost
        #Hierarchical solve_________________________________________________
        #qdot = npl.pinv(J1)*-K * err1

        #~ self.q[7:]
        #~ self.dq[6:]

        #~ Kp = K
        #~ Kd = 2*np.sqrt(Kp)

#test saturation
        #~ sat_err=0.001
        #~ err1[err1>sat_err]=sat_err
        #~ err2[err2>sat_err]=sat_err
        #~ err1[err1<-sat_err]=-sat_err
        #~ err2[err2<-sat_err]=-sat_err
        
        qddot = npl.pinv(J1)*(-1.0 * err1 -1.0 * v_err1 - 1.0*dJdq1)
        Z = null(J1)
        qddot += Z*npl.pinv(J2*Z)*(-(1.0 * err2 + 1.0 * v_err2) - J2*qddot)

        #__Saturation_______________________________________________________
        #~ qddot[qddot> 0.5]= 0.5
        #~ qddot[qddot<-0.5]=-0.5
        #__Integration______________________________________________________
        
        self.a = qddot
        self.v += np.matrix(self.a*self.dt)
        self.robot.increment(self.q, np.matrix(self.v*self.dt))
        
        #~ robotdoubleint(self.q,self.dq,qddot,self.dt)
        #~ self.v=self.dq/self.dt
        
        self.robot.display(self.q)
        self.robot.viewer.gui.refresh()
        return self.robot.com(self.q),Jcom*self.v ,errCOM,v_errCOM
        

