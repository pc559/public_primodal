'''
Evolve and save the background quantities and mode functions.
This part does not use any of the new Primodal methods---while
it is functional, it needs a rewrite.
An option would be simply getting the
time evolution of the mode functions elsewhere and plugging them
into the Primodal k-decomposition and time-integration code.
'''

try:
    from gl_c_wrapper import leggauss_c as leggauss
    ## # Python doesn't catch the failed import until the function is called.
    leggauss(3)
except:
    print('Using Numpy leggauss--- this is less accurate than QUADPTS, so be careful with convergence.')
    from numpy.polynomial.legendre import leggauss

from evol_deriv import get_spl_index,spline_eval,mode_evolution_cython
import os
import gc
import sys
import numpy as np

from scipy.integrate import odeint
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d
import math
from time import time

import config

def spline_arrs(xs,ys):
    '''Setting up splines'''
    yl = ys[:-1]
    yh = ys[1:]
    xl = np.zeros_like(yl).T
    xh = np.zeros_like(yh).T
    xl[:] = xs[:-1]
    xh[:] = xs[1:]
    xl,xh = xl.T,xh.T
    A = (yl*xh-yh*xl)/(xh-xl)
    B = (yh-yl)/(xh-xl)
    return A,B

def bar(x,k_min,k_max):
    '''Map to [-1,1]'''
    return (2*x-(k_min+k_max))/(k_max-k_min)

def unbar(x,k_min,k_max):
    '''Map to [kmin,kmax]'''
    return 0.5*((k_max-k_min)*x+k_min+k_max)

def bkgd_evolution(y, t):
    '''Evolve the background'''
    c_s = config.sound_speed(y[1],y[2],y[3])
    eps = 0.5*pow(y[2],2)/c_s
    dy = np.zeros(4)
    dy[0] = math.exp(-t)*c_s/y[3]   # tau_s
    dy[1] = y[2]                    # phi
    dy[2] = config.phi_11(y[1],y[2],y[3])  # phi'
    dy[3] = -eps*y[3]               # H
    return dy

def get_bkgd():
    '''Get the background quantities, save as interpolants.'''
    del_t = max(config.dt_pref*np.exp(config.t_init),config.early_del_t)
    del_t = min(del_t,config.late_del_t*10)
    fields = np.zeros(4)
    fields[0] = 0
    fields[1] = config.phi_0
    fields[2] = config.phi_dash_0
    fields[3] = config.H_0
    c_s_0 = config.sound_speed(fields[1],fields[2],fields[3])
    ddphi_0 = config.phi_11(fields[1],fields[2],fields[3])
    eps_s_0 = config.d_log_sound_speed(fields[1],fields[2],ddphi_0,fields[3])

    t = config.t_init
    t_target = t+del_t
    ## # [t,tau_s+...,phi,phi',H,V,dV,c_s, eps_s, eta]
    bkgd_results = [[t,*fields,config.V(fields[1]),config.V_1(fields[1]),c_s_0,eps_s_0,0]]
    while t<config.t_final:
        tol_scale_dt = np.array([1./del_t,1.,1.,1.])
        soln = odeint(bkgd_evolution,fields,[t,t_target],atol=config.bkgd_atols*tol_scale_dt,rtol=config.bkgd_rtols,full_output=0)
        fields = soln[-1]
        c_s = config.sound_speed(fields[1],fields[2],fields[3])
        ddphi = config.phi_11(fields[1],fields[2],fields[3])
        eps_s = config.d_log_sound_speed(fields[1],fields[2],ddphi,fields[3])
        epseta = config.eps_eta(fields[1],fields[2],fields[3])
        eta   = epseta/(0.5*fields[2]**2/c_s)
        bkgd_results.append([t_target,*fields,config.V(fields[1]),config.V_1(fields[1]),c_s,eps_s, eta])
        t += del_t
        del_t = max(config.dt_pref*np.exp(t),config.early_del_t)
        del_t = min(del_t,config.late_del_t*10)
        t_target = t + del_t
    c_s_f = config.sound_speed(fields[1],fields[2],fields[3])
    H_f = fields[3]
    bkgd_results = np.array(bkgd_results)
    ## # [t,tau_s,phi,phi',H,V,dV,c_s]
    bkgd_results[:,1] += -(bkgd_results[:,1][-1] + np.exp(-t)*c_s_f/H_f)
    bkgd_results = np.transpose(bkgd_results)

    ## # Setting up bkgd arrays
    ## # [t,tau_s,phi,phi',H,V,dV,c_s]
    N_array = bkgd_results[0,:]
    tau_s_array = bkgd_results[1,:]
    phi_array = bkgd_results[2,:]
    dphi_array = bkgd_results[3,:]
    H_array = bkgd_results[4,:]
    c_s_array = config.sound_speed(bkgd_results[2,:],bkgd_results[3,:],bkgd_results[4,:])
    eps_array = 0.5*bkgd_results[3,:]**2/c_s_array
    epseta_array = config.eps_eta(bkgd_results[2,:],bkgd_results[3,:],bkgd_results[4,:])
    eta_array = epseta_array/eps_array
    ddphi_array = config.phi_11(bkgd_results[2,:],bkgd_results[3,:],bkgd_results[4,:])
    eps_s_array = config.d_log_sound_speed(bkgd_results[2,:],bkgd_results[3,:],ddphi_array,bkgd_results[4,:])
    dddphi_array = config.phi_111(bkgd_results[2,:],bkgd_results[3,:],bkgd_results[4,:])
    dtau_s_array = c_s_array/(np.exp(bkgd_results[0,:])*bkgd_results[4,:])
    eps_s_dash_array = config.dd_log_sound_speed(bkgd_results[2,:],bkgd_results[3,:],ddphi_array,dddphi_array,bkgd_results[4,:],epseta_array)
    eta_dash_array = (ddphi_array**2/c_s_array + dphi_array*dddphi_array/c_s_array - dphi_array*ddphi_array*eps_s_array/c_s_array - eps_array*eta_array*eps_s_array - eps_array*eta_array**2 - eps_s_dash_array*eps_array)/eps_array
    P_array = -1.5*eta_array+3*eps_s_array+0.5*epseta_array-0.25*eta_array**2-0.5*eta_dash_array+eps_s_array*eta_array-eps_s_array**2-eps_array*eps_s_array+eps_s_dash_array
    Q_array = -2-eps_s_array
    ##########################

    interps_list = [None]*13
    interp_kind = 'linear'
    ## # [dtau_s,eps,eps_s,eta,P,Q,phi,c_s,H,tau_s,N_cross(k),N(phi)]
    interps_list[0] = interp1d(N_array,dtau_s_array,bounds_error=False,fill_value='extrapolate',kind=interp_kind)
    interps_list[1] = interp1d(N_array,eps_array,bounds_error=False,fill_value='extrapolate',kind=interp_kind)
    interps_list[2] = interp1d(N_array,eps_s_array,bounds_error=False,fill_value='extrapolate',kind=interp_kind)
    interps_list[3] = interp1d(N_array,eta_array,bounds_error=False,fill_value='extrapolate',kind=interp_kind)
    interps_list[4] = interp1d(N_array,P_array,bounds_error=False,fill_value='extrapolate',kind=interp_kind)
    interps_list[5] = interp1d(N_array,Q_array,bounds_error=False,fill_value='extrapolate',kind=interp_kind)
    interps_list[6] = interp1d(N_array,phi_array,bounds_error=False,fill_value='extrapolate',kind=interp_kind)
    interps_list[7] = interp1d(N_array,c_s_array,bounds_error=False,fill_value='extrapolate',kind=interp_kind)
    interps_list[8] = interp1d(N_array,H_array,bounds_error=False,fill_value='extrapolate',kind=interp_kind)
    interps_list[9] = interp1d(N_array,tau_s_array,bounds_error=False,fill_value='extrapolate',kind=interp_kind)
    temp_k_cross_array = np.exp(N_array)*H_array/c_s_array
    interps_list[10] = interp1d(temp_k_cross_array,N_array,bounds_error=False,fill_value='extrapolate',kind=interp_kind)
    interps_list[11] = interp1d(phi_array,N_array,bounds_error=False,fill_value='extrapolate',kind=interp_kind)
    evol_func_params = np.array([dtau_s_array,eps_array,eps_s_array,eta_array,P_array,Q_array])
    interps_list[12] = interp1d(N_array,evol_func_params,bounds_error=False,fill_value='extrapolate',kind=interp_kind)
    spls = np.array([spline_arrs(N_array,q) for q in evol_func_params])
    As,Bs = spls[:,0,:],spls[:,1,:]
    return interps_list,bkgd_results,As,Bs

def get_t_zs():
    '''Get the timesteps'''
    t = config.t_init
    t_target = t+config.early_del_t
    del_t = t_target-t
    t_list = []
    t_zs = []
    while t<config.t_final:
        t_list.append(t)
        if t_target>config.N_start_integ-config.beta_margin:
            t_zs += list(np.linspace(t,t_target,10, endpoint=True)[1:])
        t += del_t
        del_t = max(config.dt_pref*np.exp(t),config.early_del_t)
        del_t = min(del_t,config.late_del_t*10)
        t_target = t + del_t
    t_zs = np.ascontiguousarray(t_zs)
    return t_zs,t_list

def get_zks(kbars,t_zs,t_list,interps_list,As,Bs,Ns):
    '''Get the zks at each timestep.'''
    ### [dtau_s,eps,eps_s,eta,P,Q,phi,c_s,H,tau_s]
    del_t = max(config.dt_pref*np.exp(config.t_init),config.early_del_t)
    del_t = min(del_t,config.late_del_t*10)
    Nk = len(kbars)
    ks = unbar(kbars,np.exp(config.k_exp_min),np.exp(config.k_exp_max))
    k_min,k_max = np.exp(config.k_exp_min),np.exp(config.k_exp_max)

    fields = np.zeros(4*config.Nk)

    ####################################################
    ## # Set up timesteps
    t = config.t_init
    t_target = t+config.early_del_t
    del_t = t_target-t
    zs = np.zeros((len(t_zs),Nk))*1j
    dzs = np.zeros((len(t_zs),Nk))*1j

    changed = np.array([False]*len(ks)) # Must be array!
    changed_time = np.zeros(len(ks))
    set_time = np.zeros(len(ks))
    eps_0 = interps_list[1](config.t_init)
    c_s_0 = interps_list[7](config.t_init)
    H_0 = interps_list[8](config.t_init)
    odeint_time = 0
    count=0
    threshold_early = 0
    pos_re = slice(Nk)
    pos_im = slice(Nk,2*Nk)
    vel_re = slice(2*Nk,3*Nk)
    vel_im = slice(3*Nk,4*Nk)
    zs_now_temp  = np.zeros((9,Nk)).T*1j
    dzs_now_temp = np.zeros((9,Nk)).T*1j
    zs_now = np.zeros_like(zs_now_temp)*1j
    dzs_now = np.zeros_like(dzs_now_temp)*1j
    for t in t_list:
        # If set to 4.5, high k has weird glitch at switch. *Only* high k...?
        if odeint_time>0:
            # Should be set by hand to activate before deviations from SR set in.
            limit_early = math.exp(config.delta_early+t)*H_now/c_s_now
            if ks[-1]>limit_early:
                threshold_early = np.where(ks>limit_early)[0][0]
                if ('tanh' in config.label) and t>8:   #HERE
                    threshold_early = Nk
                if ('bump' in config.label) and phi_now<config.phi_f+0.5:
                    # Yes for tanh, no for reso.
                    pass
            else:
                threshold_early = Nk
            # Activates once effects from eta,eps_s etc. overtake (kc_s)**2 driving force.
            limit_late = (math.exp(config.delta_late+t)*H_now/c_s_now)
            if ks[-1]>limit_late:
                threshold_late = np.where(ks>limit_late)[0][0]
            else:
                threshold_late = Nk
            if limit_early<limit_late:
                limit_early=limit_late
        else:
            threshold_early = 0
            threshold_late = 0
            limit_late = ks[0]

        indices_to_change = np.where((changed==False)*(ks<limit_late))[0]
        #pos_re = slice(Nk)
        #pos_im = slice(Nk,2*Nk)
        #vel_re = slice(2*Nk,3*Nk)
        #vel_im = slice(3*Nk,4*Nk)
        for ind in indices_to_change:
            zeta = (fields[pos_re][ind]+1j*fields[pos_im][ind])*np.exp(-1j*ks[ind]*tau_s_now)*c_s_now/math.sqrt(2*eps_now)
            fields[pos_re][ind] = zeta.real
            fields[pos_im][ind] = zeta.imag
            dzeta = (fields[vel_re][ind]+1j*fields[vel_im][ind])*np.exp(-1j*ks[ind]*tau_s_now)*c_s_now/math.sqrt(2*eps_now)
            dzeta += -zeta*(1j*ks[ind])*(math.exp(-t)/H_now)*c_s_now
            dzeta += zeta*(-0.5*eta_now+eps_s_now)
            fields[vel_re][ind] = dzeta.real
            fields[vel_im][ind] = dzeta.imag
            changed[ind] = True
            changed_time[ind] = t

        odeint_t1 = time()
        #eval_at_times = [t,t_target]
        if t_target>config.N_start_integ-config.beta_margin:
            eval_at_times = np.linspace(t,t_target,10, endpoint=True)
        else:
            eval_at_times = np.linspace(t,t_target,2, endpoint=True)
        ## # Careful here with chaining this, don't want t twice.
        deriv_y = np.zeros_like(fields)
        full_soln = odeint(mode_evolution_cython,fields,eval_at_times,args=(ks,threshold_early,threshold_late,As,Bs,Ns,deriv_y),atol=config.zeta_atols/del_t,rtol=config.zeta_rtols)
        odeint_t2 = time()
        odeint_time += odeint_t2-odeint_t1
        fields = full_soln[-1]

        tau_s_now   = interps_list[9](t_target)
        eps_now     = interps_list[1](t_target)
        eps_s_now   = interps_list[2](t_target)
        eta_now     = interps_list[3](t_target)
        phi_now     = interps_list[6](t_target)
        c_s_now     = interps_list[7](t_target)
        H_now       = interps_list[8](t_target)

        rails_slice = slice(threshold_early,None)

        z_re = -(H_now/(math.sqrt(2*c_s_now**3)))*np.ones(len(ks[rails_slice]))
        z_im = ks[rails_slice]*c_s_now*math.exp(-t_target)/(math.sqrt(2*c_s_now**3))
        dz_re = (-eps_now-1.5*eps_s_now)*z_re
        dz_im = (-1-0.5*eps_s_now)*z_im

        fields[pos_re][rails_slice]	=  z_re
        fields[pos_im][rails_slice] 	=  z_im
        fields[vel_re][rails_slice]     =  dz_re
        fields[vel_im][rails_slice]	=  dz_im
        set_time[rails_slice] = t_target

        if t_target>config.N_start_integ-config.beta_margin:
            ## # Better for memory
            zs_now_temp[:]  = np.array(full_soln[1:,pos_re]).T+1j*np.array(full_soln[1:,pos_im]).T
            dzs_now_temp[:] = np.array(full_soln[1:,vel_re]).T+1j*np.array(full_soln[1:,vel_im]).T
            zs_now[:] = np.zeros_like(zs_now_temp)*1j
            dzs_now[:] = np.zeros_like(dzs_now_temp)*1j

            time_points    = eval_at_times[1:]
            eps_points     = interps_list[1](time_points)
            eps_s_points   = interps_list[2](time_points)
            eta_points     = interps_list[3](time_points)
            phi_points     = interps_list[6](time_points)
            c_s_points     = interps_list[7](time_points)
            H_points       = interps_list[8](time_points)
            tau_s_points   = interps_list[9](time_points)
            m_points       = np.exp(-time_points)*c_s_points*ks[changed==False][:,None]/H_points
            zs_now[changed==False] 	= zs_now_temp[changed==False]*c_s_points/np.sqrt(2*eps_points)
            dzs_now[changed==False]     = (dzs_now_temp[changed==False]-zs_now_temp[changed==False]*(1j*m_points+0.5*eta_points-eps_s_points))*c_s_points/np.sqrt(2*eps_points)
            zs_now[changed==True] 	= zs_now_temp[changed==True]*np.exp(1j*ks[changed==True][:,None]*tau_s_points)
            dzs_now[changed==True] 	= dzs_now_temp[changed==True]*np.exp(1j*ks[changed==True][:,None]*tau_s_points)

            ## # With the copy it works
            zs[count:count+len(time_points)]    = np.copy(zs_now.T)
            dzs[count:count+len(time_points)]   = np.copy(dzs_now.T)
            count   += len(time_points)

        t_temp = t + del_t
        del_t = max(config.dt_pref*np.exp(t_temp),config.early_del_t)
        del_t = min(del_t,config.late_del_t*10)
        t_target = t_temp + del_t    ## New del_t

    arr_t_zs = t_zs
    arr_zs = zs
    arr_dzs = dzs
    gc.collect()
    print("# Odeint time:",odeint_time)
    sys.stdout.flush()

    return ks,arr_zs,arr_dzs

