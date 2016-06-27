import pinocchio as se3
import numpy as np
from pinocchio.utils import *
from pinocchio.romeo_wrapper import RomeoWrapper
from pinocchio.reemc_wrapper import ReemcWrapper
from qpoases import PyQProblemB as QProblemB
from qpoases import PyQProblem as QProblem
import scipy
from IPython import embed
class PinocchioControllerAcceleration(object):
    def __init__(self,dt,robot,q_init):
        self.dt=dt
        self.robot=robot
        self.robot.viewer.gui.refresh()
        self.q =np.copy(q_init) 
        self.v =np.copy(self.robot.v0)
        self.a =np.copy(self.robot.v0)

    def controlLfRfCom(self,Lf=[.0,.0,.0],dLf=[.0,.0,.0],ddLf=[.0,.0,.0],Rf=[.0,.0,.0],dRf=[.0,.0,.0],ddRf=[.0,.0,.0],Com=[0,0,0.63],dCom=[.0,.0,.0],ddCom=[.0,.0,.0],LR=True,XYZ_RH=np.matrix([[1.],[.0],[0.8]]),FLAG_DOUBLE_SUPPORT=False):
        dLf=[.0,.0,.0]
        def errorInSE3( M,Mdes):
            '''
            Compute a 6-dim error vector (6x1 np.maptrix) caracterizing the difference
            between M and Mdes, both element of SE3.
            '''
            error = se3.log(Mdes.inverse()*M)
            return error.vector()

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

        zFeetOffset=self.robot.Mlf(self.robot.q0).translation[2]
        
        XYZ_LF=np.array(Lf)+np.array([.0,.0,zFeetOffset])
        RPY_LF=np.matrix([[.0],[.0],[.0]])
        SE3_LF=se3.SE3(self.robot.Mlf(self.robot.q0).rotation,XYZ_LF) #in case of reemc, foot orientation is not identity

        XYZ_RF=np.array(Rf)+np.array([.0,.0,zFeetOffset])#np.array([.0,.0,0.07])
        RPY_RF=np.matrix([[.0],[.0],[.0]])
        SE3_RF=se3.SE3(self.robot.Mrf(self.robot.q0).rotation,XYZ_RF)#in case of reemc, foot orientation is not identity
        
        #~ XYZ_RH=np.matrix([[1.],[.0],[0.8]])
        #~ RPY_RH=np.matrix([[.0],[.0],[.0] ])
        #~ SE3_RH=se3.SE3(self.robot.Mrh(self.robot.q0).rotation,XYZ_RH)

        #_RF________________________________________________________________
        Jrf=self.robot.Jrf(self.q).copy()
        v_ref= se3.se3.Motion(np.matrix([dRf[0],dRf[1],dRf[2],.0,.0,.0]).T)
        errRf,v_errRf,dJdqRf = errorLinkInSE3dyn(self.robot.rf,SE3_RF,v_ref,self.q,self.v)
        
        #_LF________________________________________________________________    
        Jlf=self.robot.Jlf(self.q).copy()
        v_ref= se3.se3.Motion(np.matrix([dLf[0],dLf[1],dLf[2],.0,.0,.0]).T)
        errLf,v_errLf,dJdqLf = errorLinkInSE3dyn(self.robot.lf,SE3_LF,v_ref,self.q,self.v)
        
        #_RHand_____________________________________________________________    
        #~ embed()
        #~ Jrh=self.robot.Jrh(self.q).copy()
        #~ v_ref= se3.se3.Motion(np.matrix([.0,.0,.0,.0,.0,.0]).T)
        #~ errRh,v_errRh,dJdqRh = errorLinkInSE3dyn(self.robot.rh,SE3_RH,v_ref,self.q,self.v)

        
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

        #_Post_____________________________________________________________
        errPost =   (self.q-self.robot.q0)[7:]
        v_errPost = (self.v-self.robot.v0)[6:]

        K=1000.0
        Kp_foot=K
        Kp_com=K*10.0
        Kp_Trunk=K
        Kp_post=K
        #~ Kp_Rh=.1
        

        Kd_foot= 2*np.sqrt(Kp_foot )
        Kd_com=  2*np.sqrt(Kp_com  )
        Kd_Trunk=2*np.sqrt(Kp_Trunk) 
        Kd_post= 2*np.sqrt(Kp_post ) 
        #~ Kd_Rh=   2*np.sqrt(Kp_Rh ) 
        Jpost = np.hstack( [ zero([self.robot.nv-6,6]), eye(self.robot.nv-6) ] )

        #for test, posture is included in 1st task
        eps=1e-2 #importance of posture cost

        #~ J1 = np.vstack([Jcom[:2],Jcom[2],Jlf,Jrf,JTrunk,eps*Jpost])

        if (LR):
            JflyingFoot     =      Jlf
            errflyingFoot   =    errLf
            v_errflyingFoot =  v_errLf
            dJdqFlyingFoot  =   dJdqLf
            ddFlyingFoot    =     ddLf[:2]
            
            JsupportFoot    =      Jrf
            errSupportFoot  =    errRf
            v_errSupportFoot=  v_errRf
            dJdqSupportFoot =   dJdqRf
        else:
            JflyingFoot     =      Jrf
            errflyingFoot   =    errRf
            v_errflyingFoot =  v_errRf
            dJdqFlyingFoot  =   dJdqRf
            ddFlyingFoot    =     ddRf[:2]
            
            JsupportFoot    =      Jlf
            errSupportFoot  =    errLf
            v_errSupportFoot=  v_errLf
            dJdqSupportFoot =   dJdqLf
            
        #~ embed()
        if (FLAG_DOUBLE_SUPPORT):
            #in double support, all component of the flyingfoot (witch is landed acctualy) are fixed by a PD controller
            errflyingFoot_PD   = errflyingFoot  
            v_errflyingFoot_PD = v_errflyingFoot
            dJdqFlyingFoot_PD  = dJdqFlyingFoot
            JflyingFoot_PD = JflyingFoot
        else :
            #~ #in single support, just the 3 rotations and the altitude are fixed with a PD controller 
            #~ #the rest is a coupling constrain
            errflyingFoot_PD   = errflyingFoot[2:] 
            v_errflyingFoot_PD = v_errflyingFoot[2:] 
            dJdqFlyingFoot_PD  = dJdqFlyingFoot[2:] 
            JflyingFoot_PD = JflyingFoot[2:]
            # Rq: this changes the problem size, maybe should keep a constant size with dummy line?
            
        #~ J1 = np.vstack([Jcom[2],JTrunk,JflyingFoot[2:],JsupportFoot,eps*Jpost,Jrh[:3]])
        J1 = np.vstack([Jcom[2],JTrunk,JflyingFoot_PD,JsupportFoot,eps*Jpost])
        
        Ac1 =    np.vstack([Kp_com*errComZ             - Kd_com  *v_errComZ           - dJdqCOM[2]
                          ,-Kp_Trunk*errTrunk          - Kd_Trunk*v_errTrunk          - dJdqTrunk
                          ,-Kp_foot*errflyingFoot_PD  - Kd_foot *v_errflyingFoot_PD - dJdqFlyingFoot_PD
                          ,-Kp_foot*errSupportFoot     - Kd_foot *v_errSupportFoot    - dJdqSupportFoot
                          ,eps*(-Kp_post*errPost  - Kd_post*v_errPost )                                 ])
                          #~ ,-Kp_Rh*errRh[:3]                - Kd_Rh*v_errRh[:3]                - dJdqRh[:3]  ])

        #matrix for the QP or LSQ
        self.A_FB = J1
        self.b_FB = Ac1
        #terms used for coupling
        self.JflyingFoot = JflyingFoot
        self.Jcom = Jcom
        self.dJdqCOM = dJdqCOM
        self.dJdqFlyingFoot = dJdqFlyingFoot
        return 0


