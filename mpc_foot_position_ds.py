#!/usr/bin/env python
    #Minimal PG using only steps positions as parameters
    #by using analytic solve of LIP dynamic equation
import numpy as np
import time
from IPython import embed
import matplotlib.animation as animation
import matplotlib.pyplot as plt  
import matplotlib.patches as patches
np.set_printoptions(precision=2)
np.set_printoptions(linewidth=200,  threshold=1000)


class MPC_Formulation_DS (object):
    '''
    Need to be updated!
    
    Minimal Walking Patern Genegator using only steps positions as 
    parameters by using analytic solve of LIP dynamic equation
    
    I)  computeStepsPosition(...)  
         compute $Nstep$ steps positions resulting in moving the com 
         according to a velocity command  $v$. The feet are supposed to 
         be  simple points. $Dpy$ is an heuristic on coronal plane to 
         prevent steps in line and so collision between Left and Rigth 
         foot. The sign of this heuristic is toogled at each step to
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
    
    def temporal_Sx(self,T):
        w=self.w
        return np.matrix([[np.cosh(w*T)   ,np.sinh(w*T)/w],
                           [np.sinh(w*T)*w ,np.cosh(w*T)  ]])   
    def temporal_Su(self,T):
        w=self.w
        return np.matrix([[ 1-np.cosh(w*T) ,0.   ],
                           [-w*np.sinh(w*T) ,0. ]]) 

    def temporal_Dx(self,T):
        w=self.w
        return np.matrix([[np.cosh(w*T)   ,np.sinh(w*T)/w],
                           [np.sinh(w*T)*w ,np.cosh(w*T)  ]])

    def temporal_Du(self,T):
        w=self.w
        Tds=self.Tds
        return np.matrix([[ 1-np.cosh(w*T)  + (1/Tds) * (np.sinh(w*T)/w - T) ,  (1/Tds)*(T-np.sinh(w*T)/w ) ] ,
                           [-w*np.sinh(w*T)  + (1/Tds) * (np.cosh(w*T)   - 1) ,  (1/Tds)*(1-np.cosh(w*T)  )  ]])

    def __init__ (self,Nstep=6,g=9.81,h=0.63,durrationOfStep=1.0,Dpy=0.30,weighting_Delta_px=2.0,weighting_Delta_py=2.0,weighting_p0=20.0,weighting_p1=0.0,weighting_vel=1.0,weighting_cp=100.0,Tds=0.2,pps=100):
        self.g               = g       # gravity
        self.h               = h       # com height
        self.Nstep           = Nstep   # Number of steps to predict
        self.durrationOfStep = durrationOfStep # period of one step
        self.Dpy             = Dpy     # absolute y distance from LF to RF
        self.Tds             = Tds
        self.w2              = g/h
        self.w               = np.sqrt(self.w2)
        self.pps             = pps
        self.td = durrationOfStep - self.Tds
        self.Dx=self.temporal_Dx(durrationOfStep-self.td)
        self.Du=self.temporal_Du(durrationOfStep-self.td)
        self.Sx=self.temporal_Sx(self.td)
        self.Su=self.temporal_Su(self.td)
        
        #weighting values
        self.weighting_p0       = weighting_p0         # cost on COP - center of support foot
        self.weighting_p1       = weighting_p1         # cost on foot landing position                  #WARN: This is not a constant but depend on ev
        self.weighting_vel      = weighting_vel        # cost on average velocity of com
        self.weighting_cp       = weighting_cp         # cost on capture point (pFinal - comFinal)
        self.weighting_Delta_px = weighting_Delta_px   # cost on heuristic p_i - p_(i+1)         for x
        self.weighting_Delta_py = weighting_Delta_py   # cost on heuristic p_i - p_(i+1) +/- Dpy for y
        
        
    def Fill_MPC_Matrix(self,
                                com_step_beginning=[0., 0.],
                                ev_start=0.0,
                                p0_star=[-0.001,-0.005],
                                v=[1.3,0.1],
                                x0_x=np.matrix([[0. ],[0.]]),
                                x0_y=np.matrix([[0. ],[0.]]),
                                LR=True,p1_star=[0.0,0.0],
                                weighting_p1=0.0):

        '''From an initial state x0, an evolution in this step ev, 
        some cost on LSQ Formulation (...) fill system matrix evolution'''
        
        #Print some parameters
        print(chr(27) + "[2J")
        print ('com_step_beginning {}'.format(com_step_beginning))
        print ('ev_start {}'.format(ev_start))
        print ('p0_star {}'.format(p0_star))
        print ('v {}'.format(v))
        print ('x0_x {}'.format(x0_x))
        print ('x0_y {}'.format(x0_y))
        print ('LR {}'.format(LR))
        print ('weighting_p1 {}'.format(weighting_p1))
        #~ time.sleep(0.05)

        
        #Done: work with x and y
        #Done: add toogle cost on y
        
        #Done: add cost on cop-p0
        #Done: add cost on p1-p1*

        vel_cons_x=v[0]#for x
        vel_cons_y=v[1]#for y
        
        [p0_star_x,p0_star_y]=p0_star
        [p1_star_x,p1_star_y]=p1_star
        
        #x0=x0_x
        #Load used variable in local variable
        Nstep                       =   self.Nstep
        durrationOfStep             =   self.durrationOfStep
        weighting_Delta_px          =   self.weighting_Delta_px
        weighting_Delta_py          =   self.weighting_Delta_py
        weighting_cp                =   self.weighting_cp
        weighting_vel               =   self.weighting_vel
        weighting_p0                =   self.weighting_p0         
        Dpy                         =   self.Dpy 
        if LR:
            Dpy=-Dpy
        #formulate LSq *************************************************
        (A_x,b_x) = self.fill_matrixAxbx(ev_start,x0_x)
        (A_y,b_y) = self.fill_matrixAxbx(ev_start,x0_y)
        
        (A_pos_x,b_pos_x,A_vel_x,b_vel_x) = self.extract_Apos_bpos_Avel_bvel(A_x,b_x)
        (A_pos_y,b_pos_y,A_vel_y,b_vel_y) = self.extract_Apos_bpos_Avel_bvel(A_y,b_y)
        #Express average com velocity of each step from position of com at begin and end of a step.
        #TODO: calculate velocity with x_step_beginning
        A_velAverage_x = (1./durrationOfStep)*(np.vstack([ np.zeros([1,Nstep+1]),A_pos_x[:Nstep-1] ]) - A_pos_x)
        b_velAverage_x = (1./durrationOfStep)*(np.vstack([  com_step_beginning[0],b_pos_x[:Nstep-1] ]) - b_pos_x)
        
        A_velAverage_y = (1./durrationOfStep)*(np.vstack([ np.zeros([1,Nstep+1]),A_pos_y[:Nstep-1] ]) - A_pos_y)
        b_velAverage_y = (1./durrationOfStep)*(np.vstack([  com_step_beginning[1],b_pos_y[:Nstep-1] ]) - b_pos_y)
        #Express cost matrix for capture point
        A_finalstep_x=np.zeros([1,Nstep+1])
        A_finalstep_x[0,-1]=1.0             
        b_finalstep_x=np.matrix([[0.0]])    
        
        A_finalstep_y=np.zeros([1,Nstep+1])
        A_finalstep_y[0,-1]=1.0             
        b_finalstep_y=np.matrix([[0.0]])    

        #Express cost matrix for toogle R/L foot distances
        A_p2_x = np.zeros([Nstep,Nstep+1])
        A_p2_y = np.zeros([Nstep,Nstep+1])
        for i in range(1,Nstep):
                A_p2_x[i,i]  = weighting_Delta_px
                A_p2_x[i,i+1]=-weighting_Delta_px
                A_p2_y[i,i]  = weighting_Delta_py
                A_p2_y[i,i+1]=-weighting_Delta_py   
        A_p2_x[0,1]=-weighting_Delta_px
        A_p2_y[0,1]=-weighting_Delta_py
        b_p2_x = np.zeros([Nstep,1])                
        b_p2_y = np.zeros([Nstep,1])   
        for i in range(Nstep):
            b_p2_x[i]= weighting_Delta_px *0.0
            b_p2_y[i]= weighting_Delta_py*Dpy*(-1)**i
        b_p2_x[0]+=-weighting_Delta_px*p0_star_x              
        b_p2_y[0]+=-weighting_Delta_py*p0_star_y   
        

        A_p3=np.zeros([1,Nstep+1])#p0-p0*
        A_p3[0,0]=weighting_p0
        
        b_p3_x=np.zeros([1,1])
        b_p3_y=np.zeros([1,1])
        b_p3_x[0,0]=weighting_p0*p0_star_x
        b_p3_y[0,0]=weighting_p0*p0_star_y
        
        A_p4=np.zeros([1,Nstep+1])#p1-p1*
        A_p4[0,1]=weighting_p1

        b_p4_x=np.zeros([1,1])
        b_p4_y=np.zeros([1,1])
        b_p4_x[0,0]=weighting_p1*p1_star_x
        b_p4_y[0,0]=weighting_p1*p1_star_y    
            
   
            
        #stack cost expression
        A_x=np.vstack([weighting_vel*(A_velAverage_x              ),weighting_cp*(-A_finalstep_x+A_pos_x[-1:]), A_p2_x, A_p3  , A_p4  ])
        b_x=np.vstack([weighting_vel*(-b_velAverage_x - vel_cons_x),weighting_cp*( b_finalstep_x-b_pos_x[-1:]), b_p2_x, b_p3_x, b_p4_x])
        A_y=np.vstack([weighting_vel*(A_velAverage_y              ),weighting_cp*(-A_finalstep_y+A_pos_y[-1:]), A_p2_y, A_p3  , A_p4  ])
        b_y=np.vstack([weighting_vel*(-b_velAverage_y - vel_cons_y),weighting_cp*( b_finalstep_y-b_pos_y[-1:]), b_p2_y, b_p3_y, b_p4_y])
        #stack cost expression
        #~ A_x=np.vstack([weighting_vel*(A_velAverage_x              ),weighting_cp*(-A_finalstep_x+A_pos_x[-1:]) , A_p2_x         ])
        #~ b_x=np.vstack([weighting_vel*(-b_velAverage_x - vel_cons_x),weighting_cp*( b_finalstep_x-b_pos_x[-1:]) , b_p2_x       ])
        #~ A_y=np.vstack([weighting_vel*(A_velAverage_y              ),weighting_cp*(-A_finalstep_y+A_pos_y[-1:]) , A_p2_y        ])
        #~ b_y=np.vstack([weighting_vel*(-b_velAverage_y - vel_cons_y),weighting_cp*( b_finalstep_y-b_pos_y[-1:]) , b_p2_y    ])

        #~ embed()
        A=np.vstack([     np.hstack([          A_x          ,   np.zeros(A_x.shape) ]),
                          np.hstack([ np.zeros(A_y.shape) ,            A_y          ]) ])
        b=np.vstack([b_x,b_y])
        return (A,b)
    def computePreviewOfCom(self,steps,alpha=0.0,x0=[[0,0] , [0,0]],N=20):
        return [tt, cc_x , cc_y , d_cc_x , d_cc_y]
        
    def preview_com(self,x0,p,ev_start=0.0,tn=0.0,pps=100,enable_plot=False):
        #Load used variable in local variable
        #~ Nstep           =   self.Nstep
        durrationOfStep =   self.durrationOfStep
        Tds             = self.Tds
        w2=self.w2
        #~ Dx              =   self.Dx
        #~ Du              =   self.Du
        #~ Sx              =   self.Sx
        #~ Su              =   self.Su
        #~ td              =   self.td
        temporal_Su=self.temporal_Su
        temporal_Sx=self.temporal_Sx
        temporal_Du=self.temporal_Du
        temporal_Dx=self.temporal_Dx
        #evolution in step for the first point:
        t_start=durrationOfStep*ev_start

        td = durrationOfStep - Tds

        ev_cruising_interval = np.linspace(0.0     ,1-(1.0/pps),pps)
        ev_first_interval    = np.linspace(ev_start,1-(1.0/pps),pps)
        ev_interval = ev_first_interval
        t0=t_start
        
        log_t=[]
        log_zmp=[]
        log_com=[]
        log_vcom=[]
        log_acom=[]
        it=0
        tk=0

        log_com_n  = [x0.tolist()[0]]
        log_vcom_n = [x0.tolist()[1]]
        log_t_n    = [t_start]

        flag_x0_is_in_double_support = (t_start>td)
        
        for i in range(len(p)-1):
            sum_vcom=0
            p0=p[i  ]
            p1=p[i+1]
            p01=np.matrix([[p0],
                           [p1]])

            for ev in ev_interval:
                it+=1
                t=durrationOfStep*ev
                log_t.append(t+i*durrationOfStep)
                if (t < td): #Single support phase
                    zmp=p0
                    x = temporal_Sx(t-t0) * x0  + temporal_Su(t-t0)[:,0] * p0
                else :       #double support phase
                    zmp=p0+((p1-p0)*(t-td))/Tds
                    if flag_x0_is_in_double_support : #It can append for the first step
                        x = temporal_Dx(t-t0) * x0 + (temporal_Du(t-td) - temporal_Dx(t-t0)*temporal_Du(t0-td))   * p01 #sould express analytic form of [Du(t-td)-Dx(t-t0)Du(t0-td)]
                    else: #normal case
                        x_afterSS = temporal_Sx(td-t0) * x0  + temporal_Su(td-t0)[:,0] * p0
                        x = temporal_Dx(t-td) * x_afterSS  + temporal_Du(t-td) * p01
                sum_vcom+=x.item(1)
                log_zmp.append(zmp)
                log_com.append( x.item(0))
                log_vcom.append( x.item(1))
                log_acom.append( (x.item(0) - zmp)*w2)
                
            if flag_x0_is_in_double_support :
                #~ x = temporal_Dx(durrationOfStep-t0) * x0  + temporal_Du(durrationOfStep-t0) * p01
                x = temporal_Dx(durrationOfStep-t0) * x0 + (temporal_Du(durrationOfStep-td) - temporal_Dx(durrationOfStep-t0)*temporal_Du(t0-td))   * p01
            else: #normal case
                x = temporal_Dx(durrationOfStep-td) * x_afterSS  + temporal_Du(durrationOfStep-td) * p01
            flag_x0_is_in_double_support=False
            log_com_n.append (x.tolist()[0])
            log_vcom_n.append(x.tolist()[1])
            ev_interval = ev_cruising_interval #next step start from ev=0 to ev=1(not included)
            t0=0.0
            ev=0.0
            tk+=durrationOfStep
            log_t_n.append   (tk)
            x0=x
            
        #plot :
        #~ plt.figure()
        if enable_plot :
            ax=plt.subplot(3,1,1)
            
            log_t_offset   =[t+tn for t in log_t  ]
            log_t_n_offset =[t+tn for t in log_t_n]
            
            
            plt.plot(log_t_offset,log_zmp)
            plt.plot(log_t_offset,log_com)
            plt.plot(log_t_n_offset,log_com_n,'x',lw=5,markeredgewidth=2)

            ax=plt.subplot(3,1,2)
            plt.plot(log_t_offset,log_vcom)
            plt.plot(log_t_n_offset,log_vcom_n,'x',markeredgewidth=2)
            
            ax=plt.subplot(3,1,3)
            plt.plot(log_t_offset,log_acom)
            #~ plt.plot(log_vcom,log_acom)
            #~ plt.plot(log_vcom[0],log_acom[0],'x',markeredgewidth=2)
        return [log_t,log_com,log_vcom]

    def get_linear_expression_of_next_x(self,ev_start,x0_x,x0_y,p_x,p_y):
        (a0_acomx_lip, a1_acomx_lip, b_acomx_lip) = self.get_next_x_1D(ev_start,x0_x,p_x,'coeff')
        (a0_acomy_lip, a1_acomy_lip, b_acomy_lip) = self.get_next_x_1D(ev_start,x0_y,p_y,'coeff')
        return (a0_acomx_lip, a1_acomx_lip, b_acomx_lip,a0_acomy_lip, a1_acomy_lip, b_acomy_lip)
        
    def get_next_x_1D(self,ev_start,x0,p,returnOpt='x'):
        '''get the next com and vcom dt later from ev_start according to LIP dynamic'''
        '''or get the linear expression of it'''
        #coefficient to express acc_com as a linear combination of p0 and p1
        # acc_com=a0_acom_lip*p0 + a1_acom_lip*p1 + b_acom_lip
        #This is usefull to express coupling as an equality constrain.
       
        #Load used variable in local variable
        #~ Nstep           =   self.Nstep
        durrationOfStep =   self.durrationOfStep
        Tds             =   self.Tds
        pps             =   self.pps
        #~ Dx              =   self.Dx
        #~ Du              =   self.Du
        #~ Sx              =   self.Sx
        #~ Su              =   self.Su
        #~ td              =   self.td
        w2 = self.w2
        temporal_Su=self.temporal_Su
        temporal_Sx=self.temporal_Sx
        temporal_Du=self.temporal_Du
        temporal_Dx=self.temporal_Dx
        

        a0_vcom_lip = 0.0
        a1_vcom_lip = 0.0
        b_vcom_lip  = 0.0
        
        a0_com_lip = 0.0
        a1_com_lip = 0.0
        b_com_lip  = 0.0
        
        a0_acom_lip = 0.0
        a1_acom_lip = 0.0
        b_acom_lip  = 0.0
        
        p0=p[0]
        p1=p[1]
        p01=np.matrix([[p0],
                       [p1]])    
        t_start=durrationOfStep*ev_start
        t0=t_start
        td = durrationOfStep - Tds
        t = t_start + durrationOfStep/pps
        flag_x0_is_in_double_support = (t_start>td)
        if (t < td): #Single support phase
            zmp=p0
            x = temporal_Sx(t-t0) * x0  + temporal_Su(t-t0)[:,0] * p0
            #Linear comb. coeff.
            b_vcom_lip=temporal_Sx(t-t0)[1,0]*x0[0]+temporal_Sx(t-t0)[1,1]*x0[1]
            a0_vcom_lip= temporal_Su(t-t0)[1,0]
            a1_vcom_lip= 0.0
            
            b_com_lip=temporal_Sx(t-t0)[0,0]*x0[0]+temporal_Sx(t-t0)[0,1]*x0[1]
            a0_com_lip= temporal_Su(t-t0)[0,0]
            a1_com_lip= 0.0
            
            a0_acom_lip = w2 * (a0_com_lip - 1.0)
            a1_acom_lip = w2 * (a1_com_lip )
            b_acom_lip  = w2 * b_com_lip
        else :       #double support phase
            zmp=p0+((p1-p0)*(t-td))/Tds
            if flag_x0_is_in_double_support : #It can append for the first step
                x = temporal_Dx(t-t0) * x0 + (temporal_Du(t-td) - temporal_Dx(t-t0)*temporal_Du(t0-td))   * p01 #should express analytic form of [Du(t-td)-Dx(t-t0)Du(t0-td)]
                #Linear comb. coeff.
                b_vcom_lip=temporal_Dx(t-t0)[1,0]*x0[0]+temporal_Dx(t-t0)[1,1]*x0[1]
                a0_vcom_lip=(temporal_Du(t-td) - temporal_Dx(t-t0)*temporal_Du(t0-td))[1,0]#should express analytic form of [Du(t-td)-Dx(t-t0)Du(t0-td)]
                a1_vcom_lip=(temporal_Du(t-td) - temporal_Dx(t-t0)*temporal_Du(t0-td))[1,1]#should express analytic form of [Du(t-td)-Dx(t-t0)Du(t0-td)]
                
                b_com_lip=temporal_Dx(t-t0)[0,0]*x0[0]+temporal_Dx(t-t0)[0,1]*x0[1]
                a0_com_lip=(temporal_Du(t-td) - temporal_Dx(t-t0)*temporal_Du(t0-td))[0,0]#should express analytic form of [Du(t-td)-Dx(t-t0)Du(t0-td)]
                a1_com_lip=(temporal_Du(t-td) - temporal_Dx(t-t0)*temporal_Du(t0-td))[0,1]#should express analytic form of [Du(t-td)-Dx(t-t0)Du(t0-td)]
            else: #normal case
                #~ x_afterSS = temporal_Sx(td-t0) * x0  + temporal_Su(td-t0)[:,0] * p0
                #~ x = temporal_Dx(t-td) * x_afterSS  + temporal_Du(t-td) * p01
                x = temporal_Dx(t-td)*temporal_Sx(td-t0)*x0  + (temporal_Dx(t-td)*temporal_Su(td-t0) + temporal_Du(t-td)) * p01#should express analytic form...
                #Linear comb. coeff.
                b_vcom_lip=(temporal_Dx(t-td)*temporal_Sx(td-t0))[1,0]*x0[0] + (temporal_Dx(t-td)*temporal_Sx(td-t0))[1,1]*x0[1]#should express analytic form...
                a0_vcom_lip=(temporal_Dx(t-td)*temporal_Su(td-t0) + temporal_Du(t-td))[1,0]#should express analytic form...
                a1_vcom_lip=(temporal_Dx(t-td)*temporal_Su(td-t0) + temporal_Du(t-td))[1,1]#should express analytic form...
                
                b_com_lip=(temporal_Dx(t-td)*temporal_Sx(td-t0))[0,0]*x0[0] + (temporal_Dx(t-td)*temporal_Sx(td-t0))[0,1]*x0[1]#should express analytic form...
                a0_com_lip=(temporal_Dx(t-td)*temporal_Su(td-t0) + temporal_Du(t-td))[0,0]#should express analytic form...
                a1_com_lip=(temporal_Dx(t-td)*temporal_Su(td-t0) + temporal_Du(t-td))[0,1]#should express analytic form...
                
            a0_acom_lip = w2 * (a0_com_lip + (t-td)/Tds - 1.0)
            a1_acom_lip = w2 * (a1_com_lip - (t-td)/Tds )
            b_acom_lip  = w2 * b_com_lip
                
        #~ assert (t<=durrationOfStep), "t > durrationOfStep " + str(t)
        assert ( abs(a0_com_lip *p0 + a1_com_lip *p1 + b_com_lip  - x[0]) < 1e-12),"linear expression of com mismatch com (time = " + str(t) + ')'
        assert ( abs(a0_vcom_lip*p0 + a1_vcom_lip*p1 + b_vcom_lip - x[1]) < 1e-12),"linear expression of vcom mismatch vcom (time =" + str(t) + ')'
        assert ( abs(a0_acom_lip*p0 + a1_acom_lip*p1 + b_acom_lip - (x[0]-zmp)*w2) < 1e-12),"linear expression of com mismatch com (time = " + str(t) + ')'
        print ("MPC want the com acceleration to be = {}".format((x[0]-zmp)*w2))
        if   (returnOpt=='x'):
            return x
        elif (returnOpt=='xzmp'):
            return (x,zmp)
        elif (returnOpt=='coeff'):
            return (a0_acom_lip, a1_acom_lip, b_acom_lip.item())
        
    def fill_matrixAxbx(self,ev_start,x0):
        '''
          Fill matrix A and b linearly linking com and vcom to step 
        positions.
          Given x0=[com0,Vcom0].T and ev_start, evolution in the current
        stepping phase, fill Ax and bx so that:
            
         / com1\         
        | Vcom1 |         /Step0\
        |  com2 | = [Ax].| Step1 | + (bx)
        | Vcom2 |        | Step2 |
        |  com3 |         \Step3/
         \Vcom3/
         
        All costs on evolution of com should then be expressed using Ax and bx
        '''
        #Load used variable in local variable
        Nstep           =   self.Nstep
        durrationOfStep =   self.durrationOfStep
        Dx              =   self.Dx
        Du              =   self.Du
        Sx              =   self.Sx
        Su              =   self.Su
        td              =   self.td
        temporal_Su=self.temporal_Su
        temporal_Sx=self.temporal_Sx
        temporal_Du=self.temporal_Du
        temporal_Dx=self.temporal_Dx
        
        #fill evolution matrix
        t_start=durrationOfStep*ev_start
        t0=t_start
        A_x = np.zeros([2*Nstep,Nstep+1])
        for i in range(Nstep):
             for j in range(0, i+1):
                 if (j == 0): #first colomn is a special case since the initial state may be inside a stepping phase (ev_start!=0) 
                     if (t0 < td): # two cases depending if we are in double or simple support phase
                         A_x[2*i:2*i+2,j:j+2]+=((Dx*Sx)**(i-j)*(Dx*temporal_Su(td-t0) + Du))
                     else:
                         A_x[2*i:2*i+2,j:j+2]+=((Dx*Sx)**(i-j)*(temporal_Du(durrationOfStep-td) - temporal_Dx(durrationOfStep-t0)*temporal_Du(t0-td))) #TODO: reduce computing by express analytic formulation
                 else:
                     A_x[2*i:2*i+2,j:j+2]+=((Dx*Sx)**(i-j)*(Dx*Su + Du))
        b_x =  np.matrix(np.zeros([Nstep*2,1]))
        for i in range(Nstep):
            if (t0 < td): # two cases depending if we are in double or simple support phase
                b_x[2*i:2*i+2,0]=((Dx*Sx)**i*(Dx*temporal_Sx(td-t0))*x0)
            else:
                b_x[2*i:2*i+2,0]=((Dx*Sx)**i*(temporal_Dx(durrationOfStep-t0))*x0)
        return(A_x,b_x)
        
    def extract_Apos_bpos_Avel_bvel(self,Ax,bx):
        '''split Ax and bx into position and velocity of com part'''
        #Load used variable in local variable
        Nstep           =   self.Nstep
        durrationOfStep =   self.durrationOfStep
        
        A_vel = np.zeros([Nstep,Nstep+1])
        A_pos = np.zeros([Nstep,Nstep+1])
        b_vel =  np.matrix(np.zeros([Nstep,1]))
        b_pos =  np.matrix(np.zeros([Nstep,1]))

        for i in range(Nstep):
            A_pos[i,:] = Ax[2*i  ,:]
            A_vel[i,:] = Ax[2*i+1,:]
            b_pos[i,:] = bx[2*i    ]
            b_vel[i,:] = bx[2*i+1  ]
        return (A_pos,b_pos,A_vel,b_vel)

TEST2=False
if TEST2:
    #parameters:
    Nstep=6
    g=9.81
    h=0.63
    durrationOfStep=1.0
    Dpy=0.20
    weighting_Delta_px=1.0
    weighting_Delta_py=1.0
    weighting_p0=20.0
    weighting_p1=0.0
    weighting_vel=1.0
    weighting_cp=10.0
    Tds=0.2
    pps=100
    
    com_step_beginning=[0., 0.]
    ev_start=0.0
    p0_star=[.01,.01]
    v=[1.3,0.1]
    LR=True
    p1_star=[0.0,0.0]


    
    
    
    def plot_prev_xy(p_x,p_y,x_x,x_y,fig):
        [log_t_x,log_com_x,log_vcom_x] = MPC.preview_com(x_x,p_x)
        [log_t_y,log_com_y,log_vcom_y] = MPC.preview_com(x_y,p_y)
        plt.plot(p_x,p_y)
        plt.plot(log_com_x,log_com_y)
        #ax1 = fig1.add_subplot(111, aspect='equal')
        foot_w=0.228
        foot_h=0.135
        for i in range(len(p_x)):
            ax1= plt.gca()
            ax1.add_patch(
                patches.Rectangle(
                    (p_x[i]-foot_w/2.0,p_y[i]-foot_h/2.0),   # (x,y)
                    foot_w,          # width
                    foot_h,          # height
                    fill=False,     # remove background
                )
            )
        plt.axis('equal') 
        return 0
        
    fig1 = plt.figure()
    MPC=MPC_Formulation_DS( Nstep,
                            g,
                            h,
                            durrationOfStep,
                            Dpy,
                            weighting_Delta_px,
                            weighting_Delta_py,
                            weighting_p0,
                            weighting_p1,
                            weighting_vel,
                            weighting_cp,
                            Tds,
                            pps)
    
    #From x0____________________________________________________________
    x_x=np.matrix([[0. ],[0.]])
    x_y=np.matrix([[0. ],[0.]])
    #Get MPC matrix
    (A,b) = MPC.Fill_MPC_Matrix(com_step_beginning,
                                ev_start,
                                p0_star,
                                v,
                                x_x,
                                x_y,
                                LR,
                                p1_star,
                                weighting_p1)
    #Solve
    p_opt = np.linalg.pinv(A) * b
    p=p_opt.T.tolist()[0]
    p_x=p_opt[:MPC.Nstep+1].T.tolist()[0]
    p_y=p_opt[MPC.Nstep+1:].T.tolist()[0]
    #PLOT
    [log_t,log_com_x,log_vcom_x] = MPC.preview_com(x_x,p_x,ev_start=0.0,tn=0.0,pps=100,enable_plot=False)
    [log_t,log_com_y,log_vcom_y] = MPC.preview_com(x_y,p_y,ev_start=0.0,tn=0.0,pps=100,enable_plot=True)
    
    #~ plot_prev_xy(p_x,p_y,x_x,x_y,fig1)
    
    #From x0+___________________________________________________________
    ev=0.95
    x_x=np.matrix([[log_com_x [int(ev*pps)] ],
                   [log_vcom_x[int(ev*pps)] ]])
    x_y=np.matrix([[log_com_y [int(ev*pps)] ],
                   [log_vcom_y[int(ev*pps)] ]])
    #Get MPC matrix
    (A,b) = MPC.Fill_MPC_Matrix(com_step_beginning,
                                ev,
                                p0_star,
                                v,
                                x_x,
                                x_y,
                                LR,
                                p1_star,
                                weighting_p1)
    #Solve
    p_opt = np.linalg.pinv(A) * b
    p=p_opt.T.tolist()[0]
    p_x=p_opt[:MPC.Nstep+1].T.tolist()[0]
    p_y=p_opt[MPC.Nstep+1:].T.tolist()[0]
    #PLOT
    
    [log_t,log_com_x,log_vcom_x] = MPC.preview_com(x_x,p_x,ev_start=ev,tn=0.0,pps=100,enable_plot=False)
    [log_t,log_com_y,log_vcom_y] = MPC.preview_com(x_y,p_y,ev_start=ev,tn=0.0,pps=100,enable_plot=True)
    plt.show()
    
    ev=0.0
    #Get next x get_next_x(self,ev_start,x0,p0,returnZMP=False):
    print MPC.get_linear_expression_of_next_x(ev,x_x,x_y,p_x,p_y)

TEST=False
if TEST == True:  
    def get_next_x(x0,p,ev_start=0.0,returnZMP=False):
        #coefficient to express x as a linear combination of p0 and p1
        # x=a0_vcom_lip*p0 + a1_vcom_lip*p1 + b_vcom_lip
        #This is usefull to express coupling as an equality constrain.
        a0_vcom_lip = 0.0
        a1_vcom_lip = 0.0
        b_vcom_lip  = 0.0
        
        p0=p[0]
        p1=p[1]
        p01=np.matrix([[p0],
                       [p1]])    
        t_start=durrationOfStep*ev_start
        t0=t_start
        td = durrationOfStep - Tds
        t = t_start + durrationOfStep/pps
        flag_x0_is_in_double_support = (t_start>td)
        if (t < td): #Single support phase
            zmp=p0
            x = temporal_Sx(t-t0) * x0  + temporal_Su(t-t0)[:,0] * p0
            #Linear comb. coeff.
            b_vcom_lip=temporal_Sx(t-t0)[1,0]*x0[0]+temporal_Sx(t-t0)[1,1]*x0[1]
            a0_vcom_lip= temporal_Su(t-t0)[1,0]
            a1_vcom_lip= 0.0
        else :       #double support phase
            zmp=p0+((p1-p0)*(t-td))/Tds
            if flag_x0_is_in_double_support : #It can append for the first step
                x = temporal_Dx(t-t0) * x0 + (temporal_Du(t-td) - temporal_Dx(t-t0)*temporal_Du(t0-td))   * p01 #should express analytic form of [Du(t-td)-Dx(t-t0)Du(t0-td)]
                #Linear comb. coeff.
                b_vcom_lip=temporal_Dx(t-t0)[1,0]*x0[0]+temporal_Dx(t-t0)[1,1]*x0[1]
                a0_vcom_lip=(temporal_Du(t-td) - temporal_Dx(t-t0)*temporal_Du(t0-td))[1,0]#should express analytic form of [Du(t-td)-Dx(t-t0)Du(t0-td)]
                a1_vcom_lip=(temporal_Du(t-td) - temporal_Dx(t-t0)*temporal_Du(t0-td))[1,1]#should express analytic form of [Du(t-td)-Dx(t-t0)Du(t0-td)]
            else: #normal case
                #~ x_afterSS = temporal_Sx(td-t0) * x0  + temporal_Su(td-t0)[:,0] * p0
                #~ x = temporal_Dx(t-td) * x_afterSS  + temporal_Du(t-td) * p01
                x = temporal_Dx(t-td)*temporal_Sx(td-t0)*x0  + (temporal_Dx(t-td)*temporal_Su(td-t0) + temporal_Du(t-td)) * p01#should express analytic form...
                #Linear comb. coeff.
                b_vcom_lip=(temporal_Dx(t-td)*temporal_Sx(td-t0))[1,0]*x0[0] + (temporal_Dx(t-td)*temporal_Sx(td-t0))[1,1]*x0[1]#should express analytic form...
                a0_vcom_lip=(temporal_Dx(t-td)*temporal_Su(td-t0) + temporal_Du(t-td))[1,0]#should express analytic form...
                a1_vcom_lip=(temporal_Dx(t-td)*temporal_Su(td-t0) + temporal_Du(t-td))[1,1]#should express analytic form...

                
        assert (t<=durrationOfStep), "t > durrationOfStep" + str(t)
        assert ( abs(a0_vcom_lip*p0 + a1_vcom_lip*p1 + b_vcom_lip - x[1]) < 1e-12),"linear expression of x mismatch x" + str(t)
        
        
        if returnZMP:
            return (x,zmp)
        else:
            return x
    def plot_com_ev(x0,p,ev_start=0.0,tn=0.0):
        #evolution in step for the first point:
        t_start=durrationOfStep*ev_start

        td = durrationOfStep - Tds
        
        ev_cruising_interval = np.linspace(0.0     ,1-(1.0/pps),pps)
        ev_first_interval    = np.linspace(ev_start,1-(1.0/pps),pps)
        ev_interval = ev_first_interval
        t0=t_start
        
        log_t=[]
        log_zmp=[]
        log_com=[]
        log_vcom=[]
        log_acom=[]
        it=0
        tk=0

        log_com_n  = [x0.tolist()[0]]
        log_vcom_n = [x0.tolist()[1]]
        log_t_n    = [t_start]

        flag_x0_is_in_double_support = (t_start>td)
        
        for i in range(len(p)-1):
            sum_vcom=0
            p0=p[i  ]
            p1=p[i+1]

            p01=np.matrix([[p0],
                           [p1]])    
                
                
            for ev in ev_interval:
                it+=1
                t=durrationOfStep*ev
                log_t.append(t+i*durrationOfStep)
                if (t < td): #Single support phase
                    zmp=p0
                    x = temporal_Sx(t-t0) * x0  + temporal_Su(t-t0)[:,0] * p0
                else :       #double support phase
                    zmp=p0+((p1-p0)*(t-td))/Tds
                    if flag_x0_is_in_double_support : #It can append for the first step
                        x = temporal_Dx(t-t0) * x0 + (temporal_Du(t-td) - temporal_Dx(t-t0)*temporal_Du(t0-td))   * p01 #sould express analytic form of [Du(t-td)-Dx(t-t0)Du(t0-td)]
                    else: #normal case
                        x_afterSS = temporal_Sx(td-t0) * x0  + temporal_Su(td-t0)[:,0] * p0
                        x = temporal_Dx(t-td) * x_afterSS  + temporal_Du(t-td) * p01
                sum_vcom+=x.item(1)
                log_zmp.append(zmp)
                log_com.append( x.item(0))
                log_vcom.append(x.item(1))
                log_acom.append( (x.item(0) - zmp)*w2)
                
            if flag_x0_is_in_double_support :
                #~ x = temporal_Dx(durrationOfStep-t0) * x0  + temporal_Du(durrationOfStep-t0) * p01
                x = temporal_Dx(durrationOfStep-t0) * x0 + (temporal_Du(durrationOfStep-td) - temporal_Dx(durrationOfStep-t0)*temporal_Du(t0-td))   * p01
            else: #normal case
                x = temporal_Dx(durrationOfStep-td) * x_afterSS  + temporal_Du(durrationOfStep-td) * p01
            flag_x0_is_in_double_support=False
            log_com_n.append (x.tolist()[0])
            log_vcom_n.append(x.tolist()[1])
            ev_interval = ev_cruising_interval #next step start from ev=0 to ev=1(not included)
            t0=0.0
            ev=0.0
            tk+=durrationOfStep
            log_t_n.append   (tk)
            x0=x
            
        #plot :
        #~ plt.figure()
        ax=plt.subplot(3,1,1)
        
        log_t_offset   =[t+tn for t in log_t  ]
        log_t_n_offset =[t+tn for t in log_t_n]
        
        
        plt.plot(log_t_offset,log_zmp)
        plt.plot(log_t_offset,log_com)
        plt.plot(log_t_n_offset,log_com_n,'x',lw=5,markeredgewidth=2)

        ax=plt.subplot(3,1,2)

        plt.plot(log_t_offset,log_vcom)
        plt.plot(log_t_n_offset,log_vcom_n,'x',markeredgewidth=2)
        
        ax=plt.subplot(3,1,3)
        plt.plot(log_t_offset,log_acom)
        #~ plt.plot(log_vcom,log_acom)
        #~ plt.plot(log_vcom[0],log_acom[0],'x',markeredgewidth=2)
        return [log_t,log_com,log_vcom]
        
    def temporal_Sx(T):
        return np.matrix([[np.cosh(w*T)   ,np.sinh(w*T)/w],
                           [np.sinh(w*T)*w ,np.cosh(w*T)  ]])   
    def temporal_Su(T):
        return np.matrix([[ 1-np.cosh(w*T) ,0.   ],
                           [-w*np.sinh(w*T) ,0. ]]) 

    def temporal_Dx(T):
        return np.matrix([[np.cosh(w*T)   ,np.sinh(w*T)/w],
                           [np.sinh(w*T)*w ,np.cosh(w*T)  ]])
    def temporal_Du(T):
        return np.matrix([[ 1-np.cosh(w*T)  + (1/Tds) * (np.sinh(w*T)/w - T) ,  (1/Tds)*(T-np.sinh(w*T)/w ) ] ,
                           [-w*np.sinh(w*T)  + (1/Tds) * (np.cosh(w*T)   - 1) ,  (1/Tds)*(1-np.cosh(w*T)  )  ]])
                  
    def fill_matrixAxbx(ev_start,x0):
        '''
          Fill matrix A and b linearly linking com and vcom to step 
        positions.
          Given x0=[com0,Vcom0].T and ev_start, evolution in the current
        stepping phase, fill Ax and bx so that:
            
         / com1\         
        | Vcom1 |         /Step0\
        |  com2 | = [Ax].| Step1 | + (bx)
        | Vcom2 |        | Step2 |
        |  com3 |         \Step3/
         \Vcom3/
         
        All costs on evolution of com should then be expressed using Ax and bx
        '''
        t_start=durrationOfStep*ev_start
        t0=t_start
        A_x = np.zeros([2*Nstep,Nstep+1])
        for i in range(Nstep):
            for j in range(0, i+1):
                if (j == 0): #first colomn is a special case since the initial state may be inside a stepping phase (ev_start!=0) 
                    if (t0 < td): # two cases depending if we are in double or simple support phase
                        A_x[2*i:2*i+2,j:j+2]+=((Dx*Sx)**(i-j)*(Dx*temporal_Su(td-t0) + Du))
                    else:
                        A_x[2*i:2*i+2,j:j+2]+=((Dx*Sx)**(i-j)*(temporal_Du(durrationOfStep-td) - temporal_Dx(durrationOfStep-t0)*temporal_Du(t0-td))) #TODO: reduce computing by express analytic formulation
                else:
                    A_x[2*i:2*i+2,j:j+2]+=((Dx*Sx)**(i-j)*(Dx*Su + Du))
                    
        b_x =  np.matrix(np.zeros([Nstep*2,1]))
        for i in range(Nstep):
            if (t0 < td): # two cases depending if we are in double or simple support phase
                b_x[2*i:2*i+2,0]=((Dx*Sx)**i*(Dx*temporal_Sx(td-t0))*x0)
            else:
                b_x[2*i:2*i+2,0]=((Dx*Sx)**i*(temporal_Dx(durrationOfStep-t0))*x0)
        return(A_x,b_x)
    
    def extract_Apos_bpos_Avel_bvel(Ax,bx):
        '''split Ax and bx into position and velocity of com part'''
        A_vel = np.zeros([Nstep,Nstep+1])
        A_pos = np.zeros([Nstep,Nstep+1])
        b_vel =  np.matrix(np.zeros([Nstep,1]))
        b_pos =  np.matrix(np.zeros([Nstep,1]))

        for i in range(Nstep):
            A_pos[i,:] = Ax[2*i  ,:]
            A_vel[i,:] = Ax[2*i+1,:]
            b_pos[i,:] = bx[2*i    ]
            b_vel[i,:] = bx[2*i+1  ]
        return (A_pos,b_pos,A_vel,b_vel)

    #simulate evolution of 4 entire step:
    Nstep=4
    pps = 100
    com0= 1.0
    vcom0=-0.1
    durrationOfStep=0.8
    Tds = 0.2
    vel_cons = 0.3*np.ones([Nstep,1])
    
    g=9.81
    h=0.63
    w2= g/h
    w = np.sqrt(w2)
    x0=np.matrix([[com0 ],
                  [vcom0]])
                  
    td = durrationOfStep - Tds

    Dx=temporal_Dx(durrationOfStep-td)
    Du=temporal_Du(durrationOfStep-td)
    Sx=temporal_Sx(td)
    Su=temporal_Su(td)

    #~ p0=p[0]
    ##
    # For memory:
    # x1 = (Dx*Sx)**1 * x0 + (Dx*Sx)**0*(Dx*Su + Du) * p01
    # x2 = (Dx*Sx)**2 * x0 + (Dx*Sx)**1*(Dx*Su + Du) * p01 + (Dx*Sx)**0*( Dx*Su + Du ) * p12
    # x3 = (Dx*Sx)**3 * x0 + (Dx*Sx)**2*(Dx*Su + Du) * p01 + (Dx*Sx)**1*( Dx*Su + Du ) * p12  + (Dx*Sx)**0*( Dx*Su + Du ) * p23
    # x4 = (Dx*Sx)**4 * x0 + (Dx*Sx)**3*(Dx*Su + Du) * p01 + (Dx*Sx)**2*( Dx*Su + Du ) * p12  + (Dx*Sx)**1*( Dx*Su + Du ) * p23 + (Dx*Sx)**0*( Dx*Su + Du ) * p34

    (A_x,b_x) = fill_matrixAxbx(0.0,x0)
    (A_pos,b_pos,A_vel,b_vel) = extract_Apos_bpos_Avel_bvel(A_x,b_x)

    #~ A=A_p1
    #~ b=-b_p1
    #~ p_opt = np.linalg.pinv(A) * b
    #~ p=p_opt.T.tolist()[0]
    #~ plot_com(x0,p)

    #~ A=A_pos
    #~ b=-b_pos
    #~ p_opt = np.linalg.pinv(A) * b
    #~ p=p_opt.T.tolist()[0]
    #~ plot_com(x0,p)

    #~ A= A_vel
    #~ b=-b_vel+vel_cons
    #~ p_opt = np.linalg.pinv(A) * b
    #~ p=p_opt.T.tolist()[0]
    #~ plot_com(x0,p)

    #Express average com velocity of each step from position of com at begin and end of a step.
    A_velAverage = (1./durrationOfStep)*(np.vstack([ np.zeros([1,Nstep+1]),A_pos[:Nstep-1] ]) - A_pos)
    b_velAverage = (1./durrationOfStep)*(np.vstack([  x0[0],b_pos[:Nstep-1] ]) - b_pos)
    
    #Express matrix for capture point
    A_finalstep=np.zeros([1,Nstep+1])
    A_finalstep[0,-1]=1.0
    b_finalstep=np.matrix([[0.0]])

    #stack for quadratic cost
    #~ A=np.vstack([A_velAverage            ,  A_vel[-1:]])
    #~ b=np.vstack([-b_velAverage - vel_cons, -b_vel[-1:]])
    
    A=np.vstack([A_velAverage            ,-A_finalstep+A_pos[-1:]])
    b=np.vstack([-b_velAverage - vel_cons, b_finalstep-b_pos[-1:]])

    #Find LSq solution
    p_opt = np.linalg.pinv(A) * b
    p=p_opt.T.tolist()[0]
    
    #Plot solution
    plot_com_ev(x0,p,0.0)
    

    #Do some test on preview traj from different starting time
    #(A start condition taken from the original preview, with same vector p, should lead to the same com trajectory)
    #from 0
    [log_t,log_com,log_vcom] = plot_com_ev(x0,p,0.0)
    #~ plt.figure()
    #from 0.1
    xi=np.matrix([[log_com [int(0.1*pps)] ],
                  [log_vcom[int(0.1*pps)] ]])
    plot_com_ev(xi,p,0.1)
    #from 0.5
    xi=np.matrix([[log_com [int(0.5*pps)]],
                  [log_vcom[int(0.5*pps)]]])
    plot_com_ev(xi,p,0.5)
    
    #from 0.9
    xi=np.matrix([[log_com [int(0.9*pps)]],
                  [log_vcom[int(0.9*pps)]]])
    plot_com_ev(xi,p,0.9)
    print "average velocity is {} m/s".format(np.average(log_vcom))

    #Do some test on MPC from different starting time
    #(A start condition taken from the original preview, with same cost*, should lead to the same p vector, and so same trajectory)
    #  * cost on velocity is calculated using com0 the memorised com position at the last start of step sequence.
    #~ embed()
    ev_start=0.6
    
    xi=np.matrix([[log_com [int(ev_start*pps)] ],
                  [log_vcom[int(ev_start*pps)] ]])
    
    (A_xi,b_xi) = fill_matrixAxbx(ev_start,xi)
    (A_ipos,b_ipos,A_ivel,b_ivel) = extract_Apos_bpos_Avel_bvel(A_xi,b_xi)
    
    np.matrix(A_xi)*np.matrix(p).T+np.matrix(b_xi)

    print 'p ={0}'.format(p)
    print 'x calculated from ev=0'
    print (np.matrix(A_x)*np.matrix(p).T+np.matrix(b_x))
    print 'x calculated from ev={}'.format(ev_start)
    print (np.matrix(A_xi)*np.matrix(p).T+np.matrix(b_xi))
    plot_com_ev(xi,p,ev_start)
    plt.show()
    
    xi=x0

    #~ plt.ion()
    tn=0.0 #
    log_com=[]
    log_vcom=[]
    log_acom=[]
    log_zmp=[]
    for n in range(10):
        print n
        for ev in np.linspace(0,1-(1.0/pps),pps):
         #formulate LSq *************************************************
            (A_x,b_x) = fill_matrixAxbx(ev,xi)
            (A_pos,b_pos,A_vel,b_vel) = extract_Apos_bpos_Avel_bvel(A_x,b_x)
            #Express average com velocity of each step from position of com at begin and end of a step.
            A_velAverage = (1./durrationOfStep)*(np.vstack([ np.zeros([1,Nstep+1]),A_pos[:Nstep-1] ]) - A_pos)
            b_velAverage = (1./durrationOfStep)*(np.vstack([  x0[0],b_pos[:Nstep-1] ]) - b_pos)
            #Express matrix for capture point
            A_finalstep=np.zeros([1,Nstep+1])
            A_finalstep[0,-1]=1.0
            b_finalstep=np.matrix([[0.0]])
            #stack cost expression
            A=np.vstack([A_velAverage            , -A_finalstep+ A_pos[-1:]])
            b=np.vstack([-b_velAverage - vel_cons, b_finalstep-b_pos[-1:]])
        #Find LSq solution
            p_opt = np.linalg.pinv(A) * b
            p=p_opt.T.tolist()[0]
        #plot it
            #~ plt.clf()
            #~ plot_com_ev(xi,p,ev,tn)
            #~ plt.draw()
        #save state
            (xi,zmpi)=get_next_x(xi,p,ev,True) 
            log_com.append( xi.item(0))
            log_vcom.append(xi.item(1))
            log_acom.append((xi.item(0) - zmpi)*w2)
            log_zmp.append(zmpi)
        x0=xi
        tn+=durrationOfStep

    dt=durrationOfStep/pps
    plt.figure()
    plt.plot(log_vcom)    
    plt.plot(log_acom)
    plt.plot(log_acom)
    plt.plot( (np.array(log_vcom)[1:]-np.array(log_vcom)[:-1]) * (1/dt) )
    plt.plot( log_zmp )
    plt.plot(log_acom)
    plt.show()    

    #~ plt.figure()
    #~ plt.plot(log_com)
    #~ plt.figure()
    #~ for i in range(len(log_acom)):
        #~ plt.plot(log_acom[i],log_vcom[i],'.')
        #~ plt.draw()
    #~ plt.show()

    def update_line(num, data, line):
        line.set_data(data[...,:num])
        return line,

    fig1 = plt.figure()
    data = np.array([log_acom,log_vcom])

    #~ data = np.random.rand(2, len(log_acom))
    l, = plt.plot([], [], 'r-')
    plt.xlim(-5.5, 5.5)
    plt.ylim(0.0, 2.0)

    line_ani = animation.FuncAnimation(fig1, update_line, len(log_acom), fargs=(data, l),interval=0.1, blit=True)
    #~ line_ani.save('/home/tflayols/phase2.mp4')
    plt.show()
