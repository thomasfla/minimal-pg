import numpy as np
import scipy
import matplotlib.pyplot as plt
from IPython import embed

def null(A, eps=1e-12):
    '''Compute a base of the null space of A.'''
    u, s, vh = np.linalg.svd(A)
    padding = max(0,np.shape(A)[1]-np.shape(s)[0])
    null_mask = np.concatenate(((s <= eps), np.ones((padding,),dtype=bool)),axis=0)
    null_space = scipy.compress(null_mask, vh, axis=0)
    return scipy.transpose(null_space)
    

def LSEC(A_p,b_p,A_c,b_c):
    '''
    Solve in x:  min ||A_p.x-b_p||^2
                 st.   A_c.x=b_c
    '''
    H=2*np.dot(A_p.T,A_p)
    Z=null(A_c)
    Ht=np.dot(np.dot(Z.T,H),Z)
    Gpinv=np.linalg.pinv(A_c)
    gt=np.dot(2*np.dot(b_p.T,A_p),Z) - np.dot(b_c.T,np.dot(Gpinv.T,np.dot(H,Z)))
    z_hat=np.dot(np.linalg.pinv(Ht),gt.T)
    x=(np.dot(Gpinv,b_c)+np.dot(Z,z_hat))
    return x

def example_using_LSEC():

#find x0 x1 x2 close to 10, 10, 10 st. x1+x2 = 22


    A_p= np.matrix([[1,0,0],
                    [0,1,0],
                    [0,0,1]])

    b_p= np.matrix([[10,10,10]]).T


    A_c= np.matrix([[1,1,0],
                    [0,0,0],
                    [0,0,0]])

    b_c= np.matrix([[22,0,0]]).T
    
    print LSEC(A_p,b_p,A_c,b_c)
    
#~ example_using_LSEC()
