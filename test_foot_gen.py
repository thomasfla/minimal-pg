import numpy as np
from IPython import embed
import matplotlib.pyplot as plt
from foot_trajectory_generator import Foot_trajectory_generator
if (1==1):
    ftg = Foot_trajectory_generator(0.5,0.1)
    #testing
    x0=0.0
    dx0=0.0
    ddx0=0.0

    y0=0.0
    dy0=0.0
    ddy0=0.0

    z0=0.0
    dz0=0.0
    ddz0=0.0
    t=0.0
    dt=0.001

    xx=[x0]
    yy=[y0]
    zz=[z0]

    dxx=[dx0]
    dyy=[dy0]
    dzz=[dz0]

    ddxx=[ddx0]
    ddyy=[ddy0]
    ddzz=[ddz0]


    tt=[t]

    x1=-2.0
    y1=0.5
    xx1=[x1]
    yy1=[y1]

    gxx1=[x1]
    gyy1=[y1]

    x_int   =   x0
    dx_int  =  dx0
    ddx_int = ddx0
    
    xx_int = [x_int]
    dxx_int = [dx_int]
    ddxx_int = [ddx_int]

    while(t<0.99):
        #   f0+=(np.random.rand()-0.5)/100
        #  df0+=(np.random.rand()-0.5)/1000
        # ddf0+=(np.random.rand()-0.5)/1000
        [x0,dx0,ddx0  ,  y0,dy0,ddy0  ,  z0,dz0,ddz0 , gx1,gy1]   = ftg.get_next_foot (x0, dx0, ddx0, y0, dy0, ddy0, x1, y1, t,  1.0  ,dt)
        t+=dt
        
        #itegration
        ddx_int = ddx0
        dx_int += ddx0*dt
        x_int  += dx_int*dt
        
        #LOOP on integration:
        x0=    x_int
        dx0=  dx_int
        ddx0=ddx_int
        #~ 
        
        xx_int.append(x_int)
        
        gxx1.append(gx1)  
        gyy1.append(gy1)  
        
        xx.append(x0)  
        yy.append(y0)  
        zz.append(z0)
        dxx.append(dx0)  
        dyy.append(dy0)  
        dzz.append(dz0)
        ddxx.append(ddx0)  
        ddyy.append(ddy0)  
        ddzz.append(ddz0)  
        xx1.append(x1)
        yy1.append(y1)
        tt.append(t)  
        
        #randomly changing the goal
        if np.random.rand() > 0.98 :
            x1+=(np.random.rand()-.5)/10
        if np.random.rand() > 0.98 :
            y1+=(np.random.rand()-.5)/10
            
    plt.subplot(331)
    plt.plot(tt,xx,label="position")
    plt.plot(tt,xx_int,label="position_integrated")
    plt.plot(tt,xx1,label="goal at the end")
    plt.plot(tt,gxx1,label="avaluated target")
    plt.legend()
    plt.subplot(332)
    plt.plot(tt,dxx)
    plt.subplot(333)
    plt.plot(tt,ddxx)

    plt.subplot(334)
    plt.plot(tt,yy)
    plt.plot(tt,yy1)
    plt.plot(tt,gyy1)
    plt.subplot(335)
    plt.plot(tt,dyy)
    plt.subplot(336)
    plt.plot(tt,ddyy)

    plt.subplot(337)
    plt.plot(tt,zz)
    plt.subplot(338)
    plt.plot(tt,dzz)
    plt.subplot(339)
    plt.plot(tt,ddzz)
    plt.show()
