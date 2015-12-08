#!/usr/bin/env python
    #Minimal PG using only steps positions as parameters
    #by using analytic solve of LIP dynamic equation
import numpy as np
from IPython import embed
class PgMini (object):
    '''
    Minimal Walking Patern Genegator using only steps positions as 
    parameters by using analytic solve of LIP dynamic equation
    
    I)  computeStepsPosition(...)  
         compute $Nstep$ steps positions resulting in moving the com 
         according to a velocity command  $v$. The feet are supposed to 
         be  simple points. $Dpy$ is an heuristic on coronal plane to 
         prevent steps in line and so collision between Left and Rigth 
         feet. This sign of this heuristic is toogled at each step to
         impose Left steps to be on the left side (-$Dpy$) and right 
         steps to be on the right side (+$Dpy$). The starting side is 
         set with LR boolean flag.

    II) computePreviewOfCom(...)
         compute temporal vector for the com and his velocity. It solves 
         the temporal EDO for the $Nstep$ steps with particular $steps$,
         starting from an arbitrary evolution in the current step 
         $alpha$ (from 0.0 to 1.0). and a current com state $x0$
         
         The solution is return at a $N$ samples per steps rate
         
         This function is usefull to see the all preview but should not 
         be used in an MPC implementation since only one 
         value of the COM trajectory is needed at time 0+dt.
    '''
    def __init__ (self,Nstep=6,g=9.81,h=0.63,durrationOfStep=1.0,Dpy=0.30,beta_x=1.5,beta_y=5.0,gamma=20.0):
        self.g               = g       # gravity
        self.h               = h       # com height
        self.Nstep           = Nstep   # Number of steps to predict
        self.durrationOfStep = durrationOfStep # period of one step
        self.Dpy             = Dpy     # absolute y distance from LF to RF
        self.beta_x          = beta_x  # Gain of step placement heuristic respect (x)
        self.beta_y          = beta_y  # Gain of step placement heuristic respect (y)
        self.gamma           = gamma   # Gain on step cop-p0 cost
        
        #***************************************************************
        #coeffs for the exrpession of acceleration of com as a linear function of p0_x, p0_y (usefull for coupling MPC and FB)
        #dd_c_x = (coeff_acc_x_lin_a) * p0_x + coeff_acc_x_lin_b
        #dd_c_y = (coeff_acc_y_lin_a) * p0_y + coeff_acc_y_lin_b
        #  This coeeficients are computed when computeNextCom(...) is called
        self.coeff_acc_x_lin_a =0.0
        self.coeff_acc_x_lin_b =0.0
        self.coeff_acc_y_lin_a =0.0
        self.coeff_acc_y_lin_b =0.0
        #***************************************************************
    def computeStepsPosition(self,alpha=0.0,p0=[-0.001,-0.005],v=[1.0,0.1],x0=[[0,0] , [0,0]],LR=True,p1=[0.0,0.0],gamma2=0.0,RETURN_MATRIX=False):
        #gamma=20.0
        #gamma2=200.0
        
        #const definitions
        g               = self.g     
        h               = self.h
        Nstep           = self.Nstep
        Dpy             = self.Dpy 
        durrationOfStep = self.durrationOfStep   
        beta_x          = self.beta_x 
        beta_y          = self.beta_y 
        gamma           = self.gamma
        w2= g/h
        w = np.sqrt(w2)
        
        p0_x=p0[0]
        p0_y=p0[1]
        
        p1_x=p1[0]
        p1_y=p1[1]
        
        
        
        vx= v[0] # speed command
        vy= v[1] # speed command
        
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

        A_p3=np.zeros([1,Nstep])#p0-p0*
        A_p3[0,0]=gamma
        
        A_p4=np.zeros([1,Nstep])#p1-p1*
        A_p4[0,1]=gamma2
            
        A_p_x=np.vstack([A_p1,A_p2_x,A_p3,A_p4])
        A_p_y=np.vstack([A_p1,A_p2_y,A_p3,A_p4])

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

        b_p3_x=np.zeros([1,1])#p1-p1*
        b_p3_y=np.zeros([1,1])
        b_p3_x[0,0]=gamma*p0_x
        b_p3_y[0,0]=gamma*p0_y

        b_p4_x=np.zeros([1,1])#p1-p1*
        b_p4_y=np.zeros([1,1])
        b_p4_x[0,0]=gamma2*p1_x
        b_p4_y[0,0]=gamma2*p1_y

        b_p_x=np.vstack([b_p1_x,b_p2_x,b_p3_x,b_p4_x])
        b_p_y=np.vstack([b_p1_y,b_p2_y,b_p3_y,b_p4_y])
        if (RETURN_MATRIX==False):
            #SOLVE QP: ________________________________________________________
            p_vect_x=(np.dot(np.linalg.pinv(A_p_x),b_p_x)).T
            p_vect_y=(np.dot(np.linalg.pinv(A_p_y),b_p_y)).T 
            return [p_vect_x.tolist()[0] , p_vect_y.tolist()[0]]
        else:
            A_MPC = np.vstack([A_p_x,A_p_y])
            b_MPC = np.vstack([b_p_x,b_p_y])
            return [A_MPC,b_MPC] 
    def computePreviewOfCom(self,steps,alpha=0.0,x0=[[0,0] , [0,0]],N=20):
        '''prepare preview of the com from steps position'''
        w2= self.g/self.h
        w = np.sqrt(w2)
        durrationOfStep = self.durrationOfStep   
        x0_x=np.matrix([[x0[0][0]],
                        [x0[0][1]]])
        x0_y=np.matrix([[x0[1][0]],
                        [x0[1][1]]])
        
        cc_x=[]
        cc_y=[]
        tt=[]
        d_cc_x=[]
        d_cc_y=[]

        c0_x  =x0_x[0,0]
        c0_y  =x0_y[0,0]
        d_c0_x=x0_x[1,0]
        d_c0_y=x0_y[1,0]
        tk=durrationOfStep*alpha
        for i in range(self.Nstep):
            px=steps[0][i]
            py=steps[1][i]
            for t in np.linspace(0,durrationOfStep*(1-alpha),N):#int(N*(1-alpha))):
                c_x   =     (c0_x -px) * np.cosh(w*t) + (d_c0_x/w) * np.sinh(w*t)+px
                d_c_x =   w*(c0_x -px) * np.sinh(w*t) +     d_c0_x * np.cosh(w*t) 
                c_y   =     (c0_y -py) * np.cosh(w*t) + (d_c0_y/w) * np.sinh(w*t)+py
                d_c_y =   w*(c0_y -py) * np.sinh(w*t) +     d_c0_y * np.cosh(w*t) 
                tt.append(tk+ t)
                cc_x.append(    c_x)
                cc_y.append(    c_y)
                d_cc_x.append(d_c_x)
                d_cc_y.append(d_c_y)
            tk=tk+durrationOfStep*(1-alpha)
            alpha=0 #next steps will be complete steps
            c0_x=c_x
            c0_y=c_y
            d_c0_x=d_c_x
            d_c0_y=d_c_y
        return [tt, cc_x , cc_y , d_cc_x , d_cc_y]
    def computeNextCom(self,p0,x0=[[0,0] , [0,0]],t=0.05):
        px=p0[0]
        py=p0[1]
        '''Compute COM at time  (t  < durrationOfStep*(1-alpha)  ) This function is 
        usefull for MPC implementation '''
        
        #TODO check  t  < durrationOfStep*(1-alpha)  
        w2= self.g/self.h
        w = np.sqrt(w2)
        durrationOfStep = self.durrationOfStep   
        x0_x=np.matrix([[x0[0][0]],
                        [x0[0][1]]])

        x0_y=np.matrix([[x0[1][0]],
                        [x0[1][1]]])
        
        c0_x  =x0_x[0,0]
        c0_y  =x0_y[0,0]
        d_c0_x=x0_x[1,0]
        d_c0_y=x0_y[1,0]

        c_x   =     (c0_x -px) * np.cosh(w*t) + (d_c0_x/w) * np.sinh(w*t)+px
        d_c_x = w*(c0_x -px) * np.sinh(w*t) +     d_c0_x * np.cosh(w*t) 
        c_y   =     (c0_y -py) * np.cosh(w*t) + (d_c0_y/w) * np.sinh(w*t)+py
        d_c_y = w*(c0_y -py) * np.sinh(w*t) +     d_c0_y * np.cosh(w*t) 

        #express dd_c as a linear function of p0:
        self.coeff_acc_x_lin_a = -w2*np.cosh(w*t)
        self.coeff_acc_x_lin_b = w2*c0_x*np.cosh(w*t)+w*d_c0_x*np.sinh(w*t)
        self.coeff_acc_y_lin_a = -w2*np.cosh(w*t)
        self.coeff_acc_y_lin_b = w2*c0_y*np.cosh(w*t)+w*d_c0_y*np.sinh(w*t)
        return [c_x , c_y , d_c_x , d_c_y]



