import matplotlib.pyplot as plt   
from IPython import embed
import numpy as np

def colorize_phases(STOP_TIME,durrationOfStep,ev_foot_const):
    for phase in range(int(1+STOP_TIME/durrationOfStep)):
        t0_phase = phase*durrationOfStep
        t1_phase = (phase+ev_foot_const)*durrationOfStep
        t2_phase = (phase+1)*durrationOfStep
        plt.axvspan(t0_phase, t1_phase, color='g', alpha=0.3, lw=2) #adaptative part (foot goal can change)
        plt.axvspan(t1_phase, t2_phase, color='r', alpha=0.3, lw=2) #non adaptative part
