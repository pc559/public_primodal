'''
Set up potential functions.
'''

import math
import numpy as np

def sech(x):
    try:
        return 1.0/np.cosh(x)
    except:
        return 0

def add_bump_reso(V0,dV0,ddV0,c,phi_f,d,f):
    def bump(phi):
        return 1-(np.tanh((phi_f-phi)/d))**2
    def dbump(phi):
        return -2*np.tanh((phi_f-phi)/d)*bump(phi)*(-1./d)
    def ddbump(phi):
        bump_val = bump(phi)
        return -2*bump_val*bump_val*(-1./d)-2*np.tanh((phi_f-phi)/d)*dbump(phi)*(-1./d)

    def osc(phi):
        return c*np.sin((phi-phi_f)/f)
    def dosc(phi):
        return c*np.cos((phi-phi_f)/f)/f
    def ddosc(phi):
        return -c*np.sin((phi-phi_f)/f)/f**2

    def V(phi):
        return V0(phi)*(1.+bump(phi)*osc(phi))
    def V_1(phi):
        return dV0(phi)*(1.+bump(phi)*osc(phi))+V0(phi)*(dbump(phi)*osc(phi)+bump(phi)*dosc(phi))
    def V_11(phi):
        res = ddV0(phi)*(1.+bump(phi)*osc(phi))
        res += 2*dV0(phi)*(dbump(phi)*osc(phi)+bump(phi)*dosc(phi))
        res += V0(phi)*(ddbump(phi)*osc(phi)+2*dbump(phi)*dosc(phi)+bump(phi)*ddosc(phi))
        return res

    return V,V_1,V_11

def add_reso(V0,dV0,ddV0,b,f):
    bf = b*f
    def osc(phi):
        return b*np.sin(phi/f)
    def dosc(phi):
        return b*np.cos(phi/f)/f
    def ddosc(phi):
        return -b*np.sin(phi/f)/f**2

    def V(phi):
        return V0(phi)*(1.+osc(phi))
    def V_1(phi):
        return dV0(phi)*(1.+osc(phi))+V0(phi)*dosc(phi)
    def V_11(phi):
        res = ddV0(phi)*(1.+osc(phi))
        res += 2*dV0(phi)*dosc(phi)
        res += V0(phi)*ddosc(phi)
        return res
    return V,V_1,V_11

def add_tanh(V0,dV0,ddV0,c,phi_f,d):
    def kink(phi):
        return -c*np.tanh((phi_f-phi)/d)
    def dkink(phi):
        return c*(sech((phi_f-phi)/d)**2)/d
    def ddkink(phi):
        return -2*kink(phi)*(sech((phi_f-phi)/d)**2)/d**2

    def V(phi):
        return V0(phi)*(1.+kink(phi))
    def V_1(phi):
        return dV0(phi)*(1.+kink(phi))+V0(phi)*dkink(phi)
    def V_11(phi):
        res = ddV0(phi)*(1.+kink(phi))
        res += 2*dV0(phi)*dkink(phi)
        res += V0(phi)*ddkink(phi)
        return res

    return V,V_1,V_11

def quadratic_potential_funcs(m=1e-5):
    def V(phi):
        return 0.5*(m*phi)**2
    def V_1(phi):
        return (m**2)*phi
    def V_11(phi):
        return m**2
    return V,V_1,V_11

def dbi_IR_quadratic_potential_funcs(m=1e-5,V0=1e-20):
    qV,qdV,qddV = quadratic_potential_funcs(m)
    def V(phi):
        return V0-qV(phi)
    def V_1(phi):
        return -qdV(phi)
    def V_11(phi):
        return -qddV(phi)
    return V,V_1,V_11

def starobinsky_pot_funcs(lamb=9.82e-4,Mpl=1.):
    def V(phi):
        return lamb**4*(1-np.exp(-np.sqrt(2./3.)*phi/Mpl))**2

    def V_1(phi):
        return 2*lamb**4*(1-np.exp(-np.sqrt(2./3.)*phi/Mpl))*np.exp(-np.sqrt(2./3.)*phi/Mpl)*np.sqrt(2./3.)/Mpl

    def V_11(phi):
        V_eval = V(phi)
        dV_eval = V_1(phi)
        return (np.sqrt(3./2.)/Mpl)*(0.5*np.sqrt(2./3.)*Mpl*4*lamb**4*(2./(3.*Mpl))*np.exp(-2*np.sqrt(2./3.)*phi/Mpl)-2.*dV_eval/3.)

    return V,V_1,V_11
    

'''
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbs
sbs.set()
xs = np.linspace(16,16.5,10000)
V0,V0_1,V0_11 = quadratic_potential_funcs(m=1)
#V,V_1,V_11 = starobinsky_pot_funcs(lamb=9.82e-4,Mpl=1.)
#V,V_1,V_11 = quadratic_potential_funcs(m=1)
c = 1e-2
phi_f = 5.
d = 1e-2
f = 1e-4
#V,V_1,V_11 = add_bump_reso(V,V_1,V_11,c,phi_f,d,f)
V,V_1,V_11 = add_reso(V0,V0_1,V0_11,c/f,f)
#V,V_1,V_11 = add_tanh(V,V_1,V_11,c,phi_f,d)
Vs = [V(x) for x in xs]
V0s = [V0(x) for x in xs]
V1s = [V_1(x)*f for x in xs]
V1s = [V_1(x)*f for x in xs]
V2s = [V_11(x)*f**2 for x in xs]
plt.plot(xs,Vs,label='V')
plt.plot(xs,V0s,label='V0')
plt.plot(xs,V1s,label='dV')
plt.plot(xs,V2s,label='ddV')
plt.legend()
plt.show()
'''

