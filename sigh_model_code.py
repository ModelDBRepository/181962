# -*- coding: utf-8 -*-
"""
Model of sigh and eupnea
"""

import numpy as np
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import math
#---------------------------------------------------------------------------
#   Parameters for simulation
#---------------------------------------------------------------------------
dt=15.
time=360000
sigh_threshold=-30
eupnea_threshold=-45
transients=20000
nsteps = int(time/dt)
i_transients=int(transients/dt)
t = np.linspace(0,time,nsteps)     
#---------------------------------------------------------------------------
#   Write initial condition in a vector
#---------------------------------------------------------------------------
v_e0=-34
h_e0=0.26
l_e0=0.97
c_e0=0.008
catot_e0=0.357
s_e0=0
v_s0=-53.4
h_s0=0.72
l_s0=0.93
c_s0=0.028
catot_s0=1.13
s_s0=0

y0 = [v_e0, h_e0, l_e0, c_e0, catot_e0, s_e0, v_s0, h_s0, l_s0, c_s0, catot_s0, s_s0]

#---------------------------------------------------------------------------
#   Parameters which are different or eupnea(cell_e) and sigh(sell_s)
#---------------------------------------------------------------------------
gnaps_e=2.5
gnaps_s=1
tau_pm_e=0.0001
tau_pm_s=0.1
viha_e=-90
viha_s=-70
gsyn_e=9
gsyn_s=3
vsyn_e=0
vsyn_s=-70
#---------------------------------------------------------------------------
#   Parameters which are the same for both cell types
#---------------------------------------------------------------------------
gca=0.02
gcan=1.5
ip3=1
A=0.0005
vih=30
siha=8
gih=2
gl=2.7
vl=-60
vna=50
vh=-48
vmp=-40
sh=5
smp=-6
Cm=21
tauhb=10000
Ve=400
fi=0.0001
Vi=4
sigma=0.185
LL=0.37
P=31000
Ki=1.0
Ka=0.4
Ke=0.2
Kd=0.4
vca=150
alpha=0.055
tau_ca=150
Kcan=0.74
Vpmca=2
Kpmca=0.3
vss=-10
ss=-5
tausb=5
ksyn=1
      
#---------------------------------------------------------------------------
# Function to calculate Right Hand Side of ODE
#---------------------------------------------------------------------------
def f(y, t):
    v_e = y[0]
    h_e = y[1]
    l_e = y[2]
    c_e = y[3]
    catot_e = y[4]
    s_e = y[5]
    
    v_s = y[6]
    h_s = y[7]
    l_s = y[8]
    c_s = y[9]
    catot_s = y[10]
    s_s = y[11]    
    #---------------------------------------------------------------------------
    # EUPNIC CELL (cell_e)
    #---------------------------------------------------------------------------
   
    minfp_e=1/(1+math.exp((v_e-vmp)/smp))
    hinf_e =1/(1+math.exp((v_e-vh) /sh))
    sinf_e   =1/(1+math.exp((v_e-vss) /ss))
    ihinf_e =1/(1+math.exp((v_e-viha_e)/siha))
    
    tauh_e=tauhb/math.cosh((v_e-vh)/(2*sh))
    taus_e =tausb/math.cosh((v_e-vss)/(2*ss))
    
    I_nap_e =gnaps_e*minfp_e*h_e*(v_e-vna)
    caninf_e=1/(1+(Kcan/c_e))
    I_can_e =gcan*caninf_e*(v_e-vna)
    I_ca_e  =gca*minfp_e*(v_e-vca)
    I_l_e   =gl*(v_e-vl)
    I_syn_e =gsyn_e*s_s*(v_e-vsyn_e)
    I_h_e=gih*ihinf_e*(v_e-vih)
    
    caer_e=(catot_e-c_e)/sigma
    a_e=ip3*c_e*l_e/((ip3+Ki)*(c_e+Ka))
    jerout_e=(LL+P*math.pow(a_e,3))*(caer_e-c_e)
    jerin_e=Ve*c_e*c_e/(Ke*Ke+c_e*c_e)
    jer_e=jerout_e-jerin_e
    
    jpmin_e=-I_ca_e*alpha
    jpmout_e=Vpmca*c_e*c_e/(Kpmca*Kpmca+c_e*c_e)
    jpm_e=(jpmin_e-jpmout_e)/tau_pm_e
    
    v_e_prime= (-I_nap_e-I_ca_e-I_can_e-I_l_e-I_syn_e-I_h_e)/Cm                                 
    h_e_prime=(hinf_e-h_e)/tauh_e
    l_e_prime= A*( Kd - (c_e + Kd)*l_e )                                    
    c_e_prime= fi/Vi*(jer_e+jpm_e)								
    catot_e_prime=fi/Vi*jpm_e
    s_e_prime=((1-s_e)*sinf_e-ksyn*s_e)/taus_e           
    #---------------------------------------------------------------------------
    #  SIGH CELL (cell_s)
    #---------------------------------------------------------------------------    
    minfp_s=1/(1+math.exp((v_s-vmp)/smp))
    hinf_s =1/(1+math.exp((v_s-vh) /sh))
    sinf_s   =1/(1+math.exp((v_s-vss)/ss))
    ihinf_s =1/(1+math.exp((v_s-viha_s)/siha))
    
    tauh_s=tauhb/math.cosh((v_s-vh)/(2*sh))
    taus_s =tausb/math.cosh((v_s-vss)/(2*ss))
    
    I_nap_s=gnaps_s*minfp_s*h_s*(v_s-vna)
    I_l_s  =gl*(v_s-vl)
    caninf_s =1/(1+(Kcan/c_s))
    I_can_s=gcan*caninf_s*(v_s-vna)
    I_ca_s=gca*minfp_s*(v_s-vca)
    I_syn_s   =gsyn_s*s_e*(v_s-vsyn_s)
    I_h_s=gih*ihinf_s*(v_s-vih)
    
    caer_s=(catot_s-c_s)/sigma
    a_s=ip3*c_s*l_s/((ip3+Ki)*(c_s+Ka))
    jerout_s=(LL+P*math.pow(a_s,3))*(caer_s-c_s)
    jerin_s=Ve*c_s*c_s/(Ke*Ke+c_s*c_s)
    jer_s=jerout_s-jerin_s
    
    jpmin_s=-I_ca_s*alpha
    jpmout_s=Vpmca*c_s*c_s/(Kpmca*Kpmca+c_s*c_s)
    jpm_s=(jpmin_s-jpmout_s)/tau_pm_s
    
    v_s_prime= (-I_nap_s-I_ca_s-I_can_s-I_l_s-I_syn_s-I_h_s)/Cm                                     
    h_s_prime=(hinf_s-h_s)/tauh_s
    l_s_prime= A*( Kd - (c_s + Kd)*l_s )                                    
    c_s_prime= fi/Vi*(jer_s+jpm_s)								
    catot_s_prime=fi/Vi*jpm_s
    s_s_prime=((1-s_s)*sinf_s-ksyn*s_s)/taus_s   
                                                                   
    return [v_e_prime,h_e_prime,l_e_prime,c_e_prime,catot_e_prime, s_e_prime, v_s_prime,h_s_prime,l_s_prime,c_s_prime,catot_s_prime, s_s_prime,]  

#===============================================================================
# Solve several time before plotting to get rid of transients
#===============================================================================
fig1 = plt.figure(1)
plt.subplots_adjust(left=0.05, right=0.95,wspace=0.5, hspace=0.3, top=0.95,bottom = 0.05)
y0 = [v_e0, h_e0, l_e0, c_e0, catot_e0, s_e0, v_s0, h_s0, l_s0, c_s0, catot_s0, s_s0]    
for j in range(0,2):
    soln = odeint(f, y0, t)
    y0 =soln[-1,:]
v_e = soln[:, 0]
v_s = soln[:, 6]
vavg=(v_e+v_s)/2
y0 =soln[-1,:]
#===============================================================================
# Plot average voltage 
#===============================================================================
plt.plot(t/1000.,vavg,'k')
plt.ylim(-60,-25)
plt.xlabel('time(sec)', fontsize=16)
plt.ylabel('V(mV)',fontsize=16)

