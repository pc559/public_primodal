'''
Defines the various functions needed for the equations of motion.
'''

import numpy as np
import math

def warp_funcs(lambda_dbi):
    def f(phi):
        return lambda_dbi/phi**4
    def f_1(phi,dphi):
        return -4*lambda_dbi/phi**5
    def f_11(phi,dphi,ddphi):
        return 20*lambda_dbi/phi**6
    return f,f_1,f_11

def osc_warp_funcs(lambda_dbi,osc_ampl,inv_freq):
    def f(phi):
        return (lambda_dbi/phi**4)*(1.+osc_ampl*math.sin(phi/inv_freq))
    def f_1(phi,dphi):
        res = (-4*lambda_dbi/phi**5)*(1.+osc_ampl*math.sin(phi/inv_freq))
        res += (lambda_dbi/phi**4)*(osc_ampl*math.cos(phi/inv_freq)*dphi/inv_freq)
        return res
    def f_11(phi,dphi,ddphi):
        res = (20*lambda_dbi/phi**6)*(1.+osc_ampl*math.sin(phi/inv_freq))
        res += 2*(-4*lambda_dbi/phi**5)*osc_ampl*math.cos(phi/inv_freq)*dphi/inv_freq
        res += (lambda_dbi/phi**4)*(-osc_ampl*math.sin(phi/inv_freq)*(dphi/inv_freq)**2)
        res += (lambda_dbi/phi**4)*(osc_ampl*math.cos(phi/inv_freq)*ddphi/inv_freq)
        return res
    return f,f_1,f_11

def const_c_s_funcs(c_s_0):
    def sound_speed(phi,dphi,H):
        return c_s_0*np.ones_like(phi)
    def d_log_sound_speed(phi,dphi,ddphi,H):
        return 0.*np.ones_like(phi)
    def dd_log_sound_speed(phi,dphi,ddphi,dddphi,H,epseta):
        return 0.*np.ones_like(phi)
    return sound_speed,d_log_sound_speed,dd_log_sound_speed

def gaussian_reduction_c_s_funcs(B=-0.025,beta=55*(10**5.6),phi_0=15.64):
    def u(phi):
        return B*math.exp(-beta*(phi-phi_0)**2)
    def du(phi,dphi):
        return -2*beta*(phi-phi_0)*u(phi)*dphi
    def ddu(phi,dphi,ddphi):
        u_eval = u(phi)
        du_eval = du(phi,dphi)
        res = -2*beta*dphi*u_eval*dphi
        res += -2*beta*(phi-phi_0)*du_eval*dphi
        res += -2*beta*(phi-phi_0)*u_eval*ddphi
        return res
    def sound_speed(phi,dphi,H):
        return (1-u(phi))**-0.5
    def d_log_sound_speed(phi,dphi,ddphi,H):
        return 0.5*du(phi,dphi)/(1-u(phi))
    def dd_log_sound_speed(phi,dphi,ddphi,dddphi,H,epseta):
        u_eval = u(phi)
        return 0.5*du(phi,dphi)**2/(1-u_eval)**2+0.5*ddu(phi,dphi,ddphi)/(1-u_eval)
    return sound_speed,d_log_sound_speed,dd_log_sound_speed

def const_c_s_eom_funcs(c_s_0,V,V_1,V_11):
    sound_speed,d_log_sound_speed,dd_log_sound_speed = const_c_s_funcs(c_s_0)
    def phi_11(phi,dphi,H):
        c_s = sound_speed(phi,dphi,H)
        eps = 0.5*dphi**2/c_s
        res = 0
        res += -(3*c_s**2-eps)*dphi
        res += -V_1(phi)*c_s**3/H**2
        return res
    def eps_eta(phi,dphi,H):
        c_s = sound_speed(phi,dphi,H)
        ddphi = phi_11(phi,dphi,H)
        return dphi*ddphi/c_s
    def phi_111(phi,dphi,H):    # in the case eps_s=0
        c_s = sound_speed(phi,dphi,H)
        eps = 0.5*dphi**2/c_s
        epseta = eps_eta(phi,dphi,H)
        ddphi = phi_11(phi,dphi,H)
        res = 0
        res += +epseta*dphi
        res += -(3*c_s**2-eps)*ddphi
        res += -(V_11(phi)*dphi)*c_s**3/H**2
        res += -(V_1(phi))*(2*c_s**3*eps/H**2)
        return res
    return phi_11,phi_111,eps_eta,sound_speed,d_log_sound_speed,dd_log_sound_speed


def dbi_c_s_funcs(f,f_1,f_11,V,V_1,V_11):
    def sound_speed(phi,dphi,H):
        return np.sqrt(1-f(phi)*(dphi*H)**2)
    def d_log_sound_speed(phi,dphi,ddphi,H):
        c_s = sound_speed(phi,dphi,H)
        eps = 0.5*dphi**2/c_s
        res = 0
        res += -0.5*f_1(phi,dphi)*dphi**3*H**2/c_s**2
        res += -f(phi)*dphi*ddphi*H**2/c_s**2
        res += eps*(1.-c_s**2)/c_s**2
        return res
    def dd_log_sound_speed(phi,dphi,ddphi,dddphi,H,epseta):
        c_s = sound_speed(phi,dphi,H)
        eps = 0.5*dphi**2/c_s
        eps_s = d_log_sound_speed(phi,dphi,ddphi,H)
        ff = f(phi)
        df = f_1(phi,dphi)
        ddf = f_11(phi,dphi,ddphi)
        res = 0
        res += -0.5*ddf*dphi**4*H**2
        res += -2.5*df*dphi**2*ddphi*H**2
        res += +eps*df*dphi**3*H**2
        res += df*dphi**3*H**2*eps_s
        res += -ff*ddphi**2*H**2
        res += -ff*dphi*dddphi*H**2
        res += 2*eps*ff*dphi*ddphi*H**2
        res += 2*ff*dphi*ddphi*H**2*eps_s
        res += epseta*(1-c_s**2)
        res += -2*eps*eps_s
        return res/c_s**2
    return sound_speed,d_log_sound_speed,dd_log_sound_speed

def eom_funcs(f,f_1,f_11,V,V_1,V_11,sound_speed,d_log_sound_speed,dd_log_sound_speed):
    #sound_speed,d_log_sound_speed,dd_log_sound_speed = dbi_c_s_funcs(f,f_1,f_11,V,V_1,V_11)
    def phi_11(phi,dphi,H):
        c_s = sound_speed(phi,dphi,H)
        eps = 0.5*dphi**2/c_s
        res = 0
        res += -(3*c_s**2-eps)*dphi
        res += -1.5*f_1(phi,dphi)*dphi**2/f(phi)
        res += +f_1(phi,dphi)/(H*f(phi))**2
        res += -(V_1(phi)+f_1(phi,dphi)/f(phi)**2)*c_s**3/H**2
        return res

    def eps_eta(phi,dphi,H):
        c_s = sound_speed(phi,dphi,H)
        eps = 0.5*dphi**2/c_s
        ddphi = phi_11(phi,dphi,H)
        eps_s = d_log_sound_speed(phi,dphi,ddphi,H)
        return dphi*ddphi/c_s - eps*eps_s

    def phi_111(phi,dphi,H):
        c_s = sound_speed(phi,dphi,H)
        ddphi = phi_11(phi,dphi,H)
        eps = 0.5*dphi**2/c_s
        eps_s = d_log_sound_speed(phi,dphi,ddphi,H)
        epseta = eps_eta(phi,dphi,H)
        ff = f(phi)
        df = f_1(phi,dphi)
        ddf = f_11(phi,dphi,ddphi)
        res = 0
        res += -(6*c_s**2*eps_s-epseta)*dphi
        res += -(3*c_s**2-eps)*ddphi
        res += -1.5*dphi**3*(ff*ddf-df**2)/ff**2
        res += -3*df*dphi*ddphi/ff
        res += +ddf*dphi/(ff**2*H**2)
        res += -2*df**2*dphi/(ff**3*H**2)
        res += +2*df*eps/(ff**2*H**2)
        res += -(V_11(phi)*dphi+(ddf/ff**2-2*df**2/ff**3)*dphi)*c_s**3/H**2
        res += -(V_1(phi)+df/ff**2)*(3*c_s**3*eps_s/H**2+2*c_s**3*eps/H**2)
        return res
    #return phi_11,phi_111,eps_eta,sound_speed,d_log_sound_speed,dd_log_sound_speed
    return phi_11,phi_111,eps_eta

def eom_osc(osc_ampl,inv_freq,f,f_1,f_11,V,V_1,V_11):
    sound_speed_dbi,d_log_sound_speed_dbi,dd_log_sound_speed_dbi = dbi_c_s_funcs(f,f_1,f_11,V,V_1,V_11)
    def sound_speed(phi,dphi,H):
        return sound_speed_dbi(phi,dphi,H)*(1.+osc_ampl*math.sin(phi/inv_freq))
    def d_log_sound_speed(phi,dphi,ddphi,H):
        return d_log_sound_speed_dbi(phi,dphi,ddphi,H)+(osc_ampl*math.cos(phi/inv_freq)*dphi/inv_freq)/(1.+osc_ampl*math.sin(phi/inv_freq))
    def dd_log_sound_speed(phi,dphi,ddphi,dddphi,H,epseta):
        #ddphi = phi_11(phi,dphi,H)
        #dddphi = phi_111(phi,dphi,H)
        res = 0
        res += dd_log_sound_speed_dbi(phi,dphi,ddphi,dddphi,H,epseta)
        res += -(osc_ampl*math.sin(phi/inv_freq)*(dphi/inv_freq)**2)/(1.+osc_ampl*math.sin(phi/inv_freq))
        res += (osc_ampl*math.cos(phi/inv_freq)*(ddphi/inv_freq))/(1.+osc_ampl*math.sin(phi/inv_freq))
        res += -((osc_ampl*math.cos(phi/inv_freq)*dphi/inv_freq)**2)/(1.+osc_ampl*math.sin(phi/inv_freq))**2
        return res
    phi_11,phi_111,eps_eta = eom_funcs(f,f_1,f_11,V,V_1,V_11,sound_speed,d_log_sound_speed,dd_log_sound_speed)
    return phi_11,phi_111,eps_eta,sound_speed,d_log_sound_speed,dd_log_sound_speed

def ps_approx(lambda_dbi, V0, phi_0):
    ## # From arxiv 0605045
    return lambda_dbi*V0**2/(36.*np.pi**2*phi_0**4)

def ns_approx(lambda_dbi, V0, phi_0, c_s):
    return np.sqrt(3/(lambda_dbi*V0))*phi_0**2*(-4./phi_0-2.*(phi_0**2/c_s)*(np.sqrt(3/(lambda_dbi*V0))))

