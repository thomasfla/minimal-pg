#!/usr/bin/env python
#basic usage and benchmark:      
from minimal_pg import PgMini
import matplotlib.pyplot as plt
import numpy as np
import time


#initialisation of the pg
pg = PgMini()               

#solve and return steps placement
t0=time.time()  #(tic tac mesurement)
steps = pg.computeStepsPosition() 
print "compute time: " + str((time.time()-t0)*1e3)  + " milliseconds"

#get the COM preview
[tt, cc_x , cc_y , d_cc_x , d_cc_y] = pg.computePreviewOfCom(steps)

#get COM at a particular time value
[c_x , c_y , d_c_x , d_c_y]         = pg.computeNextCom(steps)
#plot data
plt.plot(cc_x,cc_y)
plt.hold(True)
plt.plot(steps[0],steps[1])

plt.plot(steps[0],steps[1])
plt.plot([c_x],[c_y],"D")
plt.show()
