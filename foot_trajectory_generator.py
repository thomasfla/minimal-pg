import numpy as np
from IPython import embed
import matplotlib.pyplot as plt
class Foot_trajectory_generator(object):
    def __init__(self,h=0.5,time_adaptative_disabled=0.200):
        #maximum heigth for the z coordonate
        self.h = h 
        
        #when there is less than this time for the trajectory to finish, disable adaptative (using last computed coefficients)
        #this parameter should always be a positive number less than the durration of a step
        self.time_adaptative_disabled=time_adaptative_disabled 
        
        #memory of the last coeffs
        self.lastCoeffs_x = [0.0,0.0,0.0,0.0,0.0,0.0]
        self.lastCoeffs_y = [0.0,0.0,0.0,0.0,0.0,0.0]

    '''This class provide adaptative 3d trajectory for a foot from (x0,y0) to (x1,y1) using polynoms'''
    def get_next_foot(self, x0, dx0, ddx0, y0, dy0, ddy0, x1, y1, t0 , t1 ,  dt):
        '''how to reach a foot position (here using polynomials profiles)'''
        h=self.h
        if( (t1 - t0) > self.time_adaptative_disabled ):
            #compute polynoms coefficients for x and y
            Ax5=(ddx0*t0**2 - 2*ddx0*t0*t1 - 6*dx0*t0 + ddx0*t1**2 + 6*dx0*t1 + 12*x0 - 12*x1)/(2*(t0 - t1)**2*(t0**3 - 3*t0**2*t1 + 3*t0*t1**2 - t1**3))
            Ax4=(30*t0*x1 - 30*t0*x0 - 30*t1*x0 + 30*t1*x1 - 2*t0**3*ddx0 - 3*t1**3*ddx0 + 14*t0**2*dx0 - 16*t1**2*dx0 + 2*t0*t1*dx0 + 4*t0*t1**2*ddx0 + t0**2*t1*ddx0)/(2*(t0 - t1)**2*(t0**3 - 3*t0**2*t1 + 3*t0*t1**2 - t1**3))
            Ax3=(t0**4*ddx0 + 3*t1**4*ddx0 - 8*t0**3*dx0 + 12*t1**3*dx0 + 20*t0**2*x0 - 20*t0**2*x1 + 20*t1**2*x0 - 20*t1**2*x1 + 80*t0*t1*x0 - 80*t0*t1*x1 + 4*t0**3*t1*ddx0 + 28*t0*t1**2*dx0 - 32*t0**2*t1*dx0 - 8*t0**2*t1**2*ddx0)/(2*(t0 - t1)**2*(t0**3 - 3*t0**2*t1 + 3*t0*t1**2 - t1**3))
            Ax2=-(t1**5*ddx0 + 4*t0*t1**4*ddx0 + 3*t0**4*t1*ddx0 + 36*t0*t1**3*dx0 - 24*t0**3*t1*dx0 + 60*t0*t1**2*x0 + 60*t0**2*t1*x0 - 60*t0*t1**2*x1 - 60*t0**2*t1*x1 - 8*t0**2*t1**3*ddx0 - 12*t0**2*t1**2*dx0)/(2*(t0**2 - 2*t0*t1 + t1**2)*(t0**3 - 3*t0**2*t1 + 3*t0*t1**2 - t1**3))
            Ax1=-(2*t1**5*dx0 - 2*t0*t1**5*ddx0 - 10*t0*t1**4*dx0 + t0**2*t1**4*ddx0 + 4*t0**3*t1**3*ddx0 - 3*t0**4*t1**2*ddx0 - 16*t0**2*t1**3*dx0 + 24*t0**3*t1**2*dx0 - 60*t0**2*t1**2*x0 + 60*t0**2*t1**2*x1)/(2*(t0 - t1)**2*(t0**3 - 3*t0**2*t1 + 3*t0*t1**2 - t1**3))
            Ax0= (2*x1*t0**5 - ddx0*t0**4*t1**3 - 10*x1*t0**4*t1 + 2*ddx0*t0**3*t1**4 + 8*dx0*t0**3*t1**3 + 20*x1*t0**3*t1**2 - ddx0*t0**2*t1**5 - 10*dx0*t0**2*t1**4 - 20*x0*t0**2*t1**3 + 2*dx0*t0*t1**5 + 10*x0*t0*t1**4 - 2*x0*t1**5)/(2*(t0**2 - 2*t0*t1 + t1**2)*(t0**3 - 3*t0**2*t1 + 3*t0*t1**2 - t1**3))

            Ay5=(ddy0*t0**2 - 2*ddy0*t0*t1 - 6*dy0*t0 + ddy0*t1**2 + 6*dy0*t1 + 12*y0 - 12*y1)/(2*(t0 - t1)**2*(t0**3 - 3*t0**2*t1 + 3*t0*t1**2 - t1**3))
            Ay4=(30*t0*y1 - 30*t0*y0 - 30*t1*y0 + 30*t1*y1 - 2*t0**3*ddy0 - 3*t1**3*ddy0 + 14*t0**2*dy0 - 16*t1**2*dy0 + 2*t0*t1*dy0 + 4*t0*t1**2*ddy0 + t0**2*t1*ddy0)/(2*(t0 - t1)**2*(t0**3 - 3*t0**2*t1 + 3*t0*t1**2 - t1**3))
            Ay3=(t0**4*ddy0 + 3*t1**4*ddy0 - 8*t0**3*dy0 + 12*t1**3*dy0 + 20*t0**2*y0 - 20*t0**2*y1 + 20*t1**2*y0 - 20*t1**2*y1 + 80*t0*t1*y0 - 80*t0*t1*y1 + 4*t0**3*t1*ddy0 + 28*t0*t1**2*dy0 - 32*t0**2*t1*dy0 - 8*t0**2*t1**2*ddy0)/(2*(t0 - t1)**2*(t0**3 - 3*t0**2*t1 + 3*t0*t1**2 - t1**3))
            Ay2=-(t1**5*ddy0 + 4*t0*t1**4*ddy0 + 3*t0**4*t1*ddy0 + 36*t0*t1**3*dy0 - 24*t0**3*t1*dy0 + 60*t0*t1**2*y0 + 60*t0**2*t1*y0 - 60*t0*t1**2*y1 - 60*t0**2*t1*y1 - 8*t0**2*t1**3*ddy0 - 12*t0**2*t1**2*dy0)/(2*(t0**2 - 2*t0*t1 + t1**2)*(t0**3 - 3*t0**2*t1 + 3*t0*t1**2 - t1**3))
            Ay1=-(2*t1**5*dy0 - 2*t0*t1**5*ddy0 - 10*t0*t1**4*dy0 + t0**2*t1**4*ddy0 + 4*t0**3*t1**3*ddy0 - 3*t0**4*t1**2*ddy0 - 16*t0**2*t1**3*dy0 + 24*t0**3*t1**2*dy0 - 60*t0**2*t1**2*y0 + 60*t0**2*t1**2*y1)/(2*(t0 - t1)**2*(t0**3 - 3*t0**2*t1 + 3*t0*t1**2 - t1**3))
            Ay0= (2*y1*t0**5 - ddy0*t0**4*t1**3 - 10*y1*t0**4*t1 + 2*ddy0*t0**3*t1**4 + 8*dy0*t0**3*t1**3 + 20*y1*t0**3*t1**2 - ddy0*t0**2*t1**5 - 10*dy0*t0**2*t1**4 - 20*y0*t0**2*t1**3 + 2*dy0*t0*t1**5 + 10*y0*t0*t1**4 - 2*y0*t1**5)/(2*(t0**2 - 2*t0*t1 + t1**2)*(t0**3 - 3*t0**2*t1 + 3*t0*t1**2 - t1**3))
        
            self.lastCoeffs_x=[Ax5,Ax4,Ax3,Ax2,Ax1,Ax0] #save coeffs
            self.lastCoeffs_y=[Ay5,Ay4,Ay3,Ay2,Ay1,Ay0] 
        else:
            [Ax5,Ax4,Ax3,Ax2,Ax1,Ax0] = self.lastCoeffs_x #use last coeffs
            [Ay5,Ay4,Ay3,Ay2,Ay1,Ay0] = self.lastCoeffs_y
            
        #coeficients for z (deterministe)
        Az6 =         -h/((t1/2)**3*(t1 - t1/2)**3)
        Az5=    (3*t1*h)/((t1/2)**3*(t1 - t1/2)**3)
        Az4=-(3*t1**2*h)/((t1/2)**3*(t1 - t1/2)**3)
        Az3=   (t1**3*h)/((t1/2)**3*(t1 - t1/2)**3)
        
        #get the next point
        ev=t0+dt
        
        x0  =Ax0 + Ax1*ev + Ax2*ev**2 + Ax3*ev**3 + Ax4*ev**4 + Ax5*ev**5
        dx0 =Ax1 + 2*Ax2*ev + 3*Ax3*ev**2 + 4*Ax4*ev**3 + 5*Ax5*ev**4
        ddx0=2*Ax2 + 3*2*Ax3*ev + 4*3*Ax4*ev**2 + 5*4*Ax5*ev**3 
        
        y0  =Ay0 + Ay1*ev + Ay2*ev**2 + Ay3*ev**3 + Ay4*ev**4 + Ay5*ev**5
        dy0 =Ay1 + 2*Ay2*ev + 3*Ay3*ev**2 + 4*Ay4*ev**3 + 5*Ay5*ev**4
        ddy0=2*Ay2 + 3*2*Ay3*ev + 4*3*Ay4*ev**2 + 5*4*Ay5*ev**3 
        
        z0  =    Az3*ev**3 +   Az4*ev**4 +     Az5*ev**5 +    Az6*ev**6
        dz0 =  3*Az3*ev**2 + 4*Az4*ev**3 +   5*Az5*ev**4 +  6*Az6*ev**5
        ddz0=2*3*Az3*ev+   3*4*Az4*ev**2 + 4*5*Az5*ev**3 +5*6*Az6*ev**4
        
        return [x0,dx0,ddx0  ,  y0,dy0,ddy0  ,  z0,dz0,ddz0 ] 

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

#~ while(t<0.99):
    #~ #   f0+=(np.random.rand()-0.5)/100
    #~ #  df0+=(np.random.rand()-0.5)/1000
    #~ # ddf0+=(np.random.rand()-0.5)/1000
#~ 
    #~ [x0,dx0,ddx0  ,  y0,dy0,ddy0  ,  z0,dz0,ddz0]   = ftg.get_next_foot (x0, dx0, ddx0, y0, dy0, ddy0, x1, y1, t,  1.0  ,dt)
    #~ t+=dt
    #~ xx.append(x0)  
    #~ yy.append(y0)  
    #~ zz.append(z0)
    #~ dxx.append(dx0)  
    #~ dyy.append(dy0)  
    #~ dzz.append(dz0)
    #~ ddxx.append(ddx0)  
    #~ ddyy.append(ddy0)  
    #~ ddzz.append(ddz0)  
    #~ xx1.append(x1)
    #~ yy1.append(y1)
    #~ tt.append(t)  
    #~ 
    #~ #randomly changing the goal
    #~ if np.random.rand() > 0.98 :
        #~ x1+=(np.random.rand()-.5)/10
    #~ if np.random.rand() > 0.98 :
        #~ y1+=(np.random.rand()-.5)/10
        #~ 
#~ plt.subplot(331)
#~ plt.plot(tt,xx,label="position")
#~ plt.plot(tt,xx1,label="goal at the end")
#~ plt.legend()
#~ plt.subplot(332)
#~ plt.plot(tt,dxx)
#~ plt.subplot(333)
#~ plt.plot(tt,ddxx)
#~ 
#~ plt.subplot(334)
#~ plt.plot(tt,yy)
#~ plt.plot(tt,yy1)
#~ plt.subplot(335)
#~ plt.plot(tt,dyy)
#~ plt.subplot(336)
#~ plt.plot(tt,ddyy)
#~ 
#~ plt.subplot(337)
#~ plt.plot(tt,zz)
#~ plt.subplot(338)
#~ plt.plot(tt,dzz)
#~ plt.subplot(339)
#~ plt.plot(tt,ddzz)
#~ plt.show()
