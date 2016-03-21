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
    def computeStepsPosition(self,alpha=0.0,p0=[-0.001,-0.005],v=[1.0,0.1],x0=[[0,0] , [0,0]],LR=True,p1=[0.0,0.0],gamma2=0.0):
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
        

        #create general matrix : combo of x and y problems:
        #[[A_p_x,  0  ]
        #,[  0  ,A_p_y]]
        self.A_MPC=np.vstack([np.hstack([A_p_x                ,np.zeros(A_p_x.shape)]),
                              np.hstack([np.zeros(A_p_y.shape),A_p_y                ])])
        self.b_MPC=np.vstack([b_p_x,b_p_y])
        #~ np.set_printoptions(linewidth=100)


        return 0
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

TEST=True
if TEST == True:
    
    
    def plot_com(x0,p):
        pps = 10000
        td = durrationOfStep - Tds
        log_t=[]
        log_zmp=[]
        log_com=[]
        log_vcom=[]
        
        it=0
        tk=0
    
        vcom_av=[]
        vcom_av_analyt=[]
        for i in range(4):
            sum_vcom=0
            p0=p[i  ]
            p1=p[i+1]
            p01=np.matrix([[p0],
                           [p1]])
            #~ for t in np.linspace(0,durrationOfStep,10):
            for ev in np.linspace(0,1-(1.0/pps),pps):
                it+=1
                t=durrationOfStep*ev
                log_t.append(t+i*durrationOfStep)
                if (t < td): #Single support phase
                    zmp=p0
                    x = temporal_Sx(t) * x0  + temporal_Su(t) * p0
                    #~ x_last = x
                else :       #double support phase
                    x_afterSS = temporal_Sx(td) * x0  + temporal_Su(td) * p0
                    zmp=p0+((p1-p0)*(t-td))/Tds
                    x = temporal_Dx(t-td) * x_afterSS  + temporal_Du(t-td) * p01
                sum_vcom+=x.item(1)
                log_zmp.append(zmp)
                log_com.append(x.item(0))
                log_vcom.append(x.item(1))
            x = temporal_Dx(durrationOfStep-td) * x_afterSS  + temporal_Du(durrationOfStep-td) * p01
            vcom_av.append(sum_vcom/pps)
            vcom_av_analyt.append((x.item(0)-x0.item(0))/durrationOfStep)

            ev=0.0
            tk+=durrationOfStep
            x0=x
        #plot :
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(log_t,log_zmp)
        plt.plot(log_t,log_com)
        plt.subplot(2,1,2)
        plt.plot(log_t,log_vcom)
        plt.show()
    
    g=9.81
    h=0.63
    w2= g/h
    w = np.sqrt(w2)
    
    
    p=[1.0,2.0,3.0]
    

    def temporal_Sx(T):
        return np.matrix([[np.cosh(w*T)   ,np.sinh(w*T)/w],
                           [np.sinh(w*T)*w ,np.cosh(w*T)  ]])   
    def temporal_Su(T):
        return np.matrix([[ 1-np.cosh(w*T)   ],
                           [-w*np.sinh(w*T)  ]]) 
                           
    #to be filled
    def temporal_Dx(T):
        return np.matrix([[np.cosh(w*T)   ,np.sinh(w*T)/w],
                           [np.sinh(w*T)*w ,np.cosh(w*T)  ]])
    def temporal_Du(T):
        return np.matrix([[ 1-np.cosh(w*T)  + (1/Tds) * (np.sinh(w*T)/w - T) ,  (1/Tds)*(T-np.sinh(w*T)/w ) ] ,
                           [-w*np.sinh(w*T)  + (1/Tds) * (np.cosh(w*T)   - 1) ,  (1/Tds)*(1-np.cosh(w*T)  )  ]])
                           
    
 
    #simulate evolution of 4 entire step:
    import matplotlib.pyplot as plt  
     
    vcom0=0.0
    com0=-0.175138 #Magic number
    Tds = 0.2
     
    #~ vcom0=0.0
    #~ com0=-0.1836395 #Magic number
    #~ Tds = 0.001
    
    x0=np.matrix([[com0 ],
                  [vcom0]])
    mem_x0=x0
    #~ p0=-0.03
    #~ p1=0.7
    Nstep=4
    p=[-0.2,0.2,-0.2,0.2,-0.2]
        
    durrationOfStep=0.8
    
    pps = 10000
    td = durrationOfStep - Tds
    log_t=[]
    log_zmp=[]
    log_com=[]
    log_vcom=[]
    
    it=0
    tk=0
    
    vcom_av=[]
    vcom_av_analyt=[]
    for i in range(4):
        sum_vcom=0
        p0=p[i  ]
        p1=p[i+1]
        p01=np.matrix([[p0],
                       [p1]])
        #~ for t in np.linspace(0,durrationOfStep,10):
        for ev in np.linspace(0,1-(1.0/pps),pps):
            it+=1
            t=durrationOfStep*ev
            log_t.append(t+i*durrationOfStep)
            if (t < td): #Single support phase
                zmp=p0
                x = temporal_Sx(t) * x0  + temporal_Su(t) * p0
                #~ x_last = x
            else :       #double support phase
                x_afterSS = temporal_Sx(td) * x0  + temporal_Su(td) * p0
                zmp=p0+((p1-p0)*(t-td))/Tds
                x = temporal_Dx(t-td) * x_afterSS  + temporal_Du(t-td) * p01
            sum_vcom+=x.item(1)
            log_zmp.append(zmp)
            log_com.append(x.item(0))
            log_vcom.append(x.item(1))
        x = temporal_Dx(durrationOfStep-td) * x_afterSS  + temporal_Du(durrationOfStep-td) * p01
        vcom_av.append(sum_vcom/pps)
        vcom_av_analyt.append((x.item(0)-x0.item(0))/durrationOfStep)

        ev=0.0
        tk+=durrationOfStep
        x0=x
        
    Dx=temporal_Dx(durrationOfStep-td)
    Du=temporal_Du(durrationOfStep-td)
    Sx=temporal_Sx(td)
    Su=temporal_Su(td)
    p0=p[0]
    p01=np.matrix([[p[0]],
                   [p[1]]])
                   
    p12=np.matrix([[p[1]],
                   [p[2]]])       
                         
    p23=np.matrix([[p[2]],
                   [p[3]]])       
                          
    p34=np.matrix([[p[3]],
                   [p[4]]])   
    x0=mem_x0
    
    Su = np.hstack([Su,np.matrix([[0],[0]])])
    x1 = (Dx*Sx)**1 * x0 + (Dx*Sx)**0*(Dx*Su + Du) * p01
    x2 = (Dx*Sx)**2 * x0 + (Dx*Sx)**1*(Dx*Su + Du) * p01 + (Dx*Sx)**0*( Dx*Su + Du ) * p12
    x3 = (Dx*Sx)**3 * x0 + (Dx*Sx)**2*(Dx*Su + Du) * p01 + (Dx*Sx)**1*( Dx*Su + Du ) * p12  + (Dx*Sx)**0*( Dx*Su + Du ) * p23
    x4 = (Dx*Sx)**4 * x0 + (Dx*Sx)**3*(Dx*Su + Du) * p01 + (Dx*Sx)**2*( Dx*Su + Du ) * p12  + (Dx*Sx)**1*( Dx*Su + Du ) * p23 + (Dx*Sx)**0*( Dx*Su + Du ) * p34
    
    A_p1 = np.zeros([2*Nstep,2*Nstep])
    A_p1_bis = np.zeros([2*Nstep,Nstep+1])
    
    for i in range(Nstep):
        for j in range(0, i+1):
            if (j == 0):
                #~ A_p1[i,j]=(Fx**(i-j)*Fu_tr)[1,0]
                A_p1[2*i:2*i+2,2*j:2*j+2]+=((Dx*Sx)**(i-j)*(Dx*Su + Du))
                A_p1_bis[2*i:2*i+2,j:j+2]+=((Dx*Sx)**(i-j)*(Dx*Su + Du))
            else:
                A_p1[2*i:2*i+2,2*j:2*j+2]+=((Dx*Sx)**(i-j)*(Dx*Su + Du))
                A_p1_bis[2*i:2*i+2,j:j+2]+=((Dx*Sx)**(i-j)*(Dx*Su + Du))

    b_p1 =  np.matrix(np.zeros([Nstep*2,1]))
    
    for i in range(Nstep):
        b_p1[2*i:2*i+2,0]=((Dx*Sx)**(i+1)*x0)
    pf = np.vstack([p01,p12,p23,p34])
    pf_bis = np.matrix(p)

    print x1,x2,x3,x4
    print A_p1_bis*pf_bis.T+b_p1

    #   Least Square
    A=A_p1_bis
    
    
    embed()

    
    
    b=-b_p1
    p_opt = np.linalg.pinv(A) * b
    embed()
    plot_com(x0,p_opt.T.tolist()[0])
    embed()



    plt.subplot(2,1,1)
    plt.plot(log_t,log_zmp)
    plt.plot(log_t,log_com)
    plt.subplot(2,1,2)
    plt.plot(log_t,log_vcom)

    #~ plt.figure()
    #~ plt.plot(log_t,'x')
    print "average of com velocity of com durring each step is about :"
    print vcom_av
    print vcom_av_analyt
    plt.show()
    
    pg=PgMini()
    pg.computeStepsPosition()
    print pg.A_MPC
    embed()
    
