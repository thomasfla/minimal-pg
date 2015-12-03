import pinocchio as se3
import numpy as np
from pinocchio.utils import *
from pinocchio.romeo_wrapper import RomeoWrapper
from pinocchio.reemc_wrapper import ReemcWrapper
from qpoases import PyQProblemB as QProblemB
import scipy
from IPython import embed
class PinocchioControllerAcceleration(object):
    def __init__(self,dt,robot,q_init):
        self.dt=dt
        self.robot=robot
        self.robot.viewer.gui.refresh()
        self.q =np.copy(q_init) #self.robot.q0)
        self.v =np.copy(self.robot.v0)
        self.a =np.copy(self.robot.v0)
        self.lastJcom = self.robot.Jcom(self.q) #to be del.
        
    def controlLfRfCom(self,Lf=[.0,.0,.0],dLf=[.0,.0,.0],ddLf=[.0,.0,.0],Rf=[.0,.0,.0],dRf=[.0,.0,.0],ddRf=[.0,.0,.0],Com=[0,0,0.63],dCom=[.0,.0,.0],ddCom=[.0,.0,.0]):
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
            a_tot = a_corriolis
            dJdq = a_corriolis.vector() 
            return error,v_error.vector() ,dJdq

        def null(A, eps=1e-6):#-12
            '''Compute a base of the null space of A.'''
            u, s, vh = np.linalg.svd(A)
            padding = max(0,np.shape(A)[1]-np.shape(s)[0])
            null_mask = np.concatenate(((s <= eps), np.ones((padding,),dtype=bool)),axis=0)
            null_space = scipy.compress(null_mask, vh, axis=0)
            return scipy.transpose(null_space)
            
        zFeetOffset=self.robot.Mlf(self.robot.q0).translation[2]
        
        XYZ_LF=np.array(Lf)+np.array([.0,.0,zFeetOffset])
        RPY_LF=np.matrix([[.0],[.0],[.0]])
        SE3_LF=se3.SE3(self.robot.Mlf(self.robot.q0).rotation,XYZ_LF) #in case of reemc, foot orientation is not identity

        XYZ_RF=np.array(Rf)+np.array([.0,.0,zFeetOffset])#np.array([.0,.0,0.07])
        RPY_RF=np.matrix([[.0],[.0],[.0]])
        SE3_RF=se3.SE3(self.robot.Mrf(self.robot.q0).rotation,XYZ_RF)#in case of reemc, foot orientation is not identity
        
        #_RF________________________________________________________________
        Jrf=self.robot.Jrf(self.q).copy()
        v_ref= se3.se3.Motion(np.matrix([dRf[0],dRf[1],dRf[2],.0,.0,.0]).T)
        errRf,v_errRf,dJdqRf = errorLinkInSE3dyn(self.robot.rf,SE3_RF,v_ref,self.q,self.v)
        
        #_LF________________________________________________________________    
        Jlf=self.robot.Jlf(self.q).copy()
        v_ref= se3.se3.Motion(np.matrix([dLf[0],dLf[1],dLf[2],.0,.0,.0]).T)
        errLf,v_errLf,dJdqLf = errorLinkInSE3dyn(self.robot.lf,SE3_LF,v_ref,self.q,self.v)
        #embed()
        #_COM_______________________________________________________________
        Jcom=self.robot.Jcom(self.q)

        p_com, v_com, a_com = self.robot.com(self.q,self.v,self.v*0.0)
        errCOM = (np.matrix(Com).T)-self.robot.com(self.q)
        v_errCOM= v_com - (np.matrix(dCom).T)
        
        errComZ= errCOM[2,0]
        v_errComZ= v_errCOM[2,0]
        #~ v_com = Jcom*self.v

        dJdqCOM=-a_com

        #_Trunk_____________________________________________________________
        idx_Trunk = self.robot.index('root')

        MTrunk0=self.robot.position(self.robot.q0,idx_Trunk)
        MTrunk=self.robot.position(self.q,idx_Trunk)
        JTrunk=self.robot.jacobian(self.q,idx_Trunk)[3:6]
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

        Jpost = np.hstack( [ zero([self.robot.nv-6,6]), eye(self.robot.nv-6) ] )
        errPost =   Kp_post*(self.q-self.robot.q0)[7:]
        v_errPost = Kd_post*(self.v-self.robot.v0)[6:]
        #for test, posture is included in 1st task
        eps=1e-6 #importance of posture cost

        #~ J1 =    np.vstack([Jcom[:2],Jcom[2],Jlf,Jrf,JTrunk])
        J1 =    np.vstack([Jcom[:2],Jcom[2],Jlf,Jrf,JTrunk,eps*Jpost])
        #~ embed()
        
        #~ Ac1=    np.vstack([np.matrix(ddCom).T[:2]                   - dJdqCOM[:2] 
                          #~ , Kp_com*errComZ    - Kd_com  *v_errComZ  - dJdqCOM[2]
                          #~ ,-Kp_foot*errLf     - Kd_foot *v_errLf    - dJdqLf
                          #~ ,-Kp_foot*errRf     - Kd_foot *v_errRf    - dJdqRf   
                          #~ ,-Kp_Trunk*errTrunk - Kd_Trunk*v_errTrunk - dJdqTrunk])
#~ 
        Ac1=    np.vstack([np.matrix(ddCom).T[:2]                   - dJdqCOM[:2] #x,y
                          , Kp_com*errComZ    - Kd_com  *v_errComZ  - dJdqCOM[2]  #z
                          #,-0.0*Kp_foot*errLf[:2] - 0.0*Kd_foot *v_errLf[:2] - 0.0*dJdqLf[:2]  #x,y np.matrix(ddLf).T[:2]                  +0.0* dJdqLf[:2] #
                          #,-0.0*Kp_foot*errLf[2:] - 0.0*Kd_foot *v_errLf[2:] - 0.0*dJdqLf[2:]   #z ,rx,ry,rz    
                          ,(-1.0*Kp_foot*errLf[0] - 1.0*Kd_foot *v_errLf[0] - 0.0*dJdqLf[0])  #z
                          ,(-1.0*Kp_foot*errLf[1] - 1.0*Kd_foot *v_errLf[1] - 0.0*dJdqLf[1])     #y
                          ,(-1.0*Kp_foot*errLf[2] - 1.0*Kd_foot *v_errLf[2] - 0.0*dJdqLf[2])     #x np.matrix(ddLf).T[0]  +1.0* dJdqLf[2]
                          ,(-1.0*Kp_foot*errLf[3] - 1.0*Kd_foot *v_errLf[3] - 0.0*dJdqLf[3])
                          ,(-1.0*Kp_foot*errLf[4] - 1.0*Kd_foot *v_errLf[4] - 0.0*dJdqLf[4])
                          ,(-1.0*Kp_foot*errLf[5] - 1.0*Kd_foot *v_errLf[5] - 0.0*dJdqLf[5])
                          ,-Kp_foot*errRf     - Kd_foot *v_errRf    - dJdqRf
                          ,-Kp_Trunk*errTrunk - Kd_Trunk*v_errTrunk - dJdqTrunk
                          ,eps*(-errPost  - v_errPost )             ])
                          
                          
    #_TASK2 STACK_______________________________________________________
        #_Posture___________________________________________________________
        #~ Jpost = np.hstack( [ zero([self.robot.nv-6,6]), eye(self.robot.nv-6) ] )
        #~ errPost =   Kp_post*(self.q-self.robot.q0)[7:]
        #~ v_errPost = Kd_post*(self.v-self.robot.v0)[6:]

        errpost =  -1 * (self.q-self.robot.q0)[7:]
        err2 = errPost
        v_err2 = v_errPost
        J2 = Jpost
    #Hierarchical solve_________________________________________________

        dt2=self.dt**2
        dt=self.dt
    #Using Least Squares*********************
        qddot = npl.pinv(J1)*(Ac1)
    #Using QPoases:**************************
        A=J1
        b=Ac1
        
        qpb = QProblemB(self.robot.nv)
        H=np.array( A.T*A).T
        g=np.array(-A.T*b).T[0]
        lb=-100000.0*np.ones(self.robot.nv)
        ub= 100000.0*np.ones(self.robot.nv)
        #~ embed()
        qpb.init(H,g,lb,ub,np.array([100]))
        x_hat2=np.zeros(self.robot.nv)
        qpb.getPrimalSolution(x_hat2)
        qddot = np.matrix(x_hat2).T
   #*****************************************     


        #~ Z = null(J1)

        #~ qddot += Z*npl.pinv(J2*Z)*(-(1.0 * err2 + 1.0 * v_err2) - J2*qddot) #for tests, just one task

        self.a = qddot
        self.v += np.matrix(self.a*self.dt)
        self.robot.increment(self.q, np.matrix(self.v*self.dt))

        self.robot.display(self.q)
        self.robot.viewer.gui.refresh()
        return self.robot.com(self.q),Jcom*self.v ,errCOM,v_errCOM
        


