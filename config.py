"""
Docstring
"""

import sys
import math
import numpy as np
from potential_def_arg import add_bump_reso, add_reso, add_tanh, dbi_IR_quadratic_potential_funcs, quadratic_potential_funcs, starobinsky_pot_funcs
import dbi_funcs_np

calc_bis = True

## # IJ_list = [[1,1,1],[1,1,0],[0,0,0],[1,1,0],[1,1,0],[0,0,1]]
def bkgd_coeff_loc(N, H, eps ,eta ,c_s ,eps_s ,s ,phi):
    Sigma = eps*H**2/c_s**2
    ## # DBI case
    lambda_h_int = H**2*eps*(1-c_s**2)/(2*c_s**4)
    ## # In the write-up, this corresponds to:
    ## # 0,1,2,3+4,5
    ## # s=0 is 0 for DBI
    if s==0: return -(eps*np.exp(3*N)/(H*c_s**4))*(1.-c_s**2-2*c_s**2*lambda_h_int/Sigma)*H**3
    elif s==1: return (eps*np.exp(3*N)/c_s**4)*(3.-3.*c_s**2-eps+eta)*H**2
    elif s==2: return -(eps*np.exp(1*N)/c_s**2)*(1.-c_s**2+eps+eta-2.*eps_s)
    elif s==3: return -(eps**2*np.exp(3*N)/(2*c_s**4))*(eps-4)*H**2
    elif s==4: return -(eps**3*np.exp(3*N)/(4*c_s**4))*H**2
    elif s==5: return 0
    else: return 0

## # IJ_list = [[1,1,1],[1,1,0],[0,0,0],[1,1,0],[1,1,0],[0,0,1]]
def bkgd_coeff_eql(N, H, eps, eta, c_s, eps_s, s, phi):
    Sigma = eps*H**2/c_s**2
    ## # DBI case
    lambda_h_int = H**2*eps*(1-c_s**2)/(2*c_s**4)
    ## # Cancellation for DBI, 1+2*l/S=1/cs^2
    if s==0: return (eps*np.exp(3*N)/(H*c_s**2))*(1+2*lambda_h_int/Sigma)*H**3
    elif s==1: return (eps*np.exp(3*N)/c_s**4)*(-3.*c_s**2)*H**2
    elif s==2: return -(eps*np.exp(1*N)/c_s**2)*(-c_s**2)
    elif s==3: return -(eps**2*np.exp(3*N)/(2*c_s**4))*(eps-4)*H**2
    elif s==4: return -(eps**3*np.exp(3*N)/(4*c_s**4))*H**2
    elif s==5: return -(eps*np.exp(1*N)/(H*c_s**2))*H
    else: return 0

def bkgd_coeff_2f(N,H,eps,eta,c_s,eps_s,s,phi):
    fac = -2./(H*c_s)
    u = 1.-1./c_s**2
    if s==0: return 2*fac*u*eps*np.exp(3*N)*H**2
    elif s==1: return 2*fac*2*(1-u)*eps_s*eps*np.exp(3*N)*H**2
    elif s==5: return 2*fac*u*eps*np.exp(N)
    else: return 0

def bkgd_coeff_planck_eft(N,H,eps,eta,c_s,eps_s,s,phi):
    fac = -2./(H*c_s)
    u = 1.-1./c_s**2
    c3 = (3./2)*(1-c_s**2) ## # DBI
    if s==0: return -np.exp(3*N)*u*eps*H**2/c_s**2
    elif s==5: return u*eps*np.exp(N)
    else: return 0

def eft_fnl_eql(c_s,c3):
    u = 1.-1./c_s**2
    ## # Planck typo? #return -u*(-0.275-0.0780*c_s**2-(2./3)*0.780*c3)
    return -u*(-0.275-0.0780*c_s**2-(2./3)*0.0780*c3)

def eft_fnl_ort(c_s,c3):
    u = 1.-1./c_s**2
    return -u*(0.0159-0.0167*c_s**2-(2./3)*0.0167*c3)

################################################
##   BKGD SCENARIO LABEL
perturbed_potential = False
bkgd_label = 'dbi'
#bkgd_label = 'soft_dbi'
#bkgd_label = 'malda'
#bkgd_label = 'staro'
################################################
################################################
##   FEATURE SCENARIO LABEL
feat_label = ''
#feat_label = 'reso'#_'+str(sys.argv[1])
#feat_label = 'reso_bump'#_3e-4'#+str(sys.argv[1])
#feat_label = 'tanh'#_'+str(sys.argv[1])
################################################
label = bkgd_label
if len(feat_label)>0:
    label += '_'+feat_label
################################################

################################################
##   SCENARIO PARAMETERS
param_string = ''
if bkgd_label=='dbi':
    lambda_dbi = 2.00475e15#*float(sys.argv[1])#2.21079e15*1.05127
    V0 = 5.2e-12
    ## # scale by 0.65, 2.
    beta_dbi_IR = 2.9e-1#*float(sys.argv[1])#*2.
    #label += '_'+sys.argv[1]
    f,f_1,f_11 = dbi_funcs_np.warp_funcs(lambda_dbi)
    V,V_1,V_11 = dbi_IR_quadratic_potential_funcs(np.sqrt(beta_dbi_IR*V0/3.),V0)
    param_string = "# lambda_dbi,V0,beta_IR "+str([lambda_dbi,V0,beta_dbi_IR])
    #print("# lambda_dbi,V0,beta_IR",lambda_dbi,V0,beta_dbi_IR)
elif bkgd_label=='soft_dbi':
    lambda_dbi = 1e2
    m = 6e-6
    f,f_1,f_11 = dbi_funcs_np.warp_funcs(lambda_dbi)
    V,V_1,V_11 = quadratic_potential_funcs(m)
    param_string = "# lambda_dbi,m "+str([lambda_dbi,m])
    #print("# lambda_dbi,m",lambda_dbi,m)
elif bkgd_label=='malda':
    ## # HERE!
    m = 6.38e-6#5.77e-6#6e-6/np.sqrt(1.08)
    V,V_1,V_11 = quadratic_potential_funcs(m)
    param_string = "# m = "+str(m)
    #print("# m =",m)
elif bkgd_label=='staro':
    lamb = 9.82e-4
    Mpl = 1.
    V,V_1,V_11 = starobinsky_pot_funcs(lamb,Mpl)
    param_string = "# lamb,Mpl "+str([lamb,Mpl])
    #print("# lamb,Mpl",lamb,Mpl)

if label[0:8]=='dbi_tanh':
    c = float(sys.argv[1])#1.8e-3
    d = 5e-3#2e-3#5e-3
    phi_f = 15.27+np.sqrt(2*0.007)*np.log(2.2e-2*5./d)#2e-4)#d)#14.84
    V,V_1,V_11 = add_tanh(V,V_1,V_11,c,phi_f,d)
    param_string += '\n'+' '.join(str(x) for x in ('# c,d,phi_step = ',c,d,phi_f))
    #print('# c,d,phi_step =',c,d,phi_f)
elif feat_label[0:4]=='tanh':
    c = 5e-4#100*5e-5
    try:
        c = float(sys.argv[1])
        #feat_label += 'c'+sys.argv[1]
        #label += 'c'+sys.argv[1]
    except:
        pass
    feat_label += '_c_'+str(c)
    label += '_c_'+str(c)
    d = 1e-2#5e-3#5e-3
    phi_f = 15.457#15.27+np.sqrt(2*0.007)*np.log(2.2e-2*5./d)#2e-4)#d)#14.84
    V,V_1,V_11 = add_tanh(V,V_1,V_11,c,phi_f,d)
    param_string += '\n'+' '.join(str(x) for x in ('# c,d,phi_step = ',c,d,phi_f))
    #print('# c,d,phi_step =',c,d,phi_f)
elif label[0:13]=='dbi_reso_bump':
    #phi_f = 0.5317
    phi_f = 0.53442
    d = 1e6#2*8e-3#1e6#2*2e-3*2#*1e6
    freq_inv = 0.0015*float(sys.argv[1])
    #label += '_'+sys.argv[1]
    b = 0.1*2*1e-6*freq_inv*2000#*(freq_inv/3e-4)**2.5#*0.001818*6/(10)
    feat_label += '_'+str(freq_inv)
    label += '_'+str(freq_inv)
    V,V_1,V_11 = add_bump_reso(V,V_1,V_11,b,phi_f,d,freq_inv)
    param_string += '\n'+' '.join(str(x) for x in ('# b,f,d,phi_step = ',b,freq_inv,d,phi_f))
    #print('# b,f,d,phi_step =',b,freq_inv,d,phi_f)
elif feat_label[0:9]=='reso_bump':
    phi_f = 15.25#10.3
    d = 1e-2*10
    b = 1e-2
    freq_inv = 0.0005#2e-2#float(sys.argv[1])#0.001818*6
    V,V_1,V_11 = add_bump_reso(V,V_1,V_11,b*freq_inv,phi_f,d,freq_inv)
    param_string += '\n'+' '.join(str(x) for x in ('# b,f,d,phi_step = ',b,freq_inv,d,phi_f))
    #print('# b,f,d,phi_step =',b,freq_inv,d,phi_f)
elif label[0:8]=='dbi_reso':
    freq_inv = 2e-4
    b = 2*1e-6
    feat_label += '_'+str(freq_inv)
    label += '_'+str(freq_inv)
    V,V_1,V_11 = add_reso(V,V_1,V_11,b,freq_inv)
    param_string += '\n'+' '.join(str(x) for x in ('# b,f = ',b,freq_inv))
    '''b = 1e-3
    freq_inv = float(sys.argv[1])#6e-4#6.125e-4
    #mu3 = (6e-4)**3
    V,V_1,V_11 = add_reso(V,V_1,V_11,b,freq_inv)
    param_string += '\n'+' '.join(str(x) for x in ('# b,f = ',b,freq_inv))'''
    #print('# b,f =',b,freq_inv)
elif feat_label[0:4]=='reso':
    b = 1e-3
    freq_inv = 0.005#0.0545#0.5*1.5*2e-2*1000./550.
    feat_label += '_'+str(freq_inv)
    label += '_'+str(freq_inv)
    #mu3 = (6e-4)**3
    V,V_1,V_11 = add_reso(V,V_1,V_11,b*freq_inv,freq_inv)
    param_string += '\n'+' '.join(str(x) for x in ('# b,f = ',b,freq_inv))
    #print('# b,f =',b,freq_inv)
################################################

################################################
## DEFINING SOUND SPEED, PHI'' AND DERIVS
if 'dbi' in bkgd_label:
    sound_speed,d_log_sound_speed,dd_log_sound_speed = dbi_funcs_np.dbi_c_s_funcs(f,f_1,f_11,V,V_1,V_11) 
    phi_11,phi_111,eps_eta = dbi_funcs_np.eom_funcs(f,f_1,f_11,V,V_1,V_11,sound_speed,d_log_sound_speed,dd_log_sound_speed)
else:
    c_s_0 =1.
    phi_11,phi_111,eps_eta,sound_speed,d_log_sound_speed,dd_log_sound_speed = dbi_funcs_np.const_c_s_eom_funcs(c_s_0,V,V_1,V_11)
################################################

################################################
## SET t_init, t_final, bkgd_coeff, shape_indices
if label[0:3]=='dbi':
    t_init = -6.
    t_final = 18.
    #bkgd_coeff = bkgd_coeff_planck_eft
    #shape_indices = [0,5]
    bkgd_coeff = bkgd_coeff_eql
    shape_indices = [0,1,2,3,4,5]
else:
    t_init = -6.#0.
    t_final = 18.#20.
    bkgd_coeff = bkgd_coeff_loc
    shape_indices = [0,1,2,3,4]
shape_indices = np.array(shape_indices, dtype=np.int32)
################################################

################################################
## INITIAL CONDITIONS
if bkgd_label=='malda' or bkgd_label=='staro':
    phi_0 = 16.5
    if label=='staro':
        phi_0 = 5.
    c_s_0 = sound_speed(0,0,0)
    phi_dash_0 = -V_1(phi_0)*c_s_0/V(phi_0)
    eps_0 = 0.5*phi_dash_0**2
    H_0 = math.sqrt(V(phi_0)/(3.-eps_0))
elif bkgd_label=='soft_dbi':
    phi_0 = 16.5
    c_s_0 = math.sqrt(1./(1+f(phi_0)*V_1(phi_0)**2/(3.*V(phi_0))))
    phi_dash_0 = -V_1(phi_0)*c_s_0/V(phi_0)
    H_0 = math.sqrt((1./(f(phi_0)*c_s_0)+V(phi_0)-1./f(phi_0))/3.)
    phi_dash_0 = -math.sqrt((1-c_s_0**2)/(H_0**2*f(phi_0)))
elif label[0:3]=='dbi':
    phi_0 = 0.46042
    c_s_0 = math.sqrt(1./(1+f(phi_0)*V_1(phi_0)**2/(3.*V(phi_0))))
    H_0 = math.sqrt((1./(f(phi_0)*c_s_0)+V(phi_0)-1./f(phi_0))/3.)
    # NEGATIVE
    phi_dash_0 = -1*-math.sqrt((1-c_s_0**2)/(H_0**2*f(phi_0)))
################################################

################################################
##   SET k RANGE
k_exp_max   = np.log(1.2*2500./14370.634)
k_exp_min   = -np.log(1000.)+k_exp_max
k_min,k_max = np.exp(k_exp_min),np.exp(k_exp_max)
k_pivot     = 0.05
################################################

################################################
##   SET METHOD PARAMS
l_max = 30-3#55-3
n_s_mod = 0.9649-1
## # OVERRIDE BELOW WITH LOG BASIS
logbasis = True
##############################################
if n_s_mod!=0:
    working_basis_params = [k_min, k_max, l_max, 1+n_s_mod, True, False, None]
    final_basis_params   = [k_min, k_max, l_max+3, n_s_mod, True, False, None]
else:
    working_basis_params = [k_min, k_max, l_max,  0., True, False, None, True]
    final_basis_params   = [k_min, k_max, l_max+3, 0., True, False, None]
##############################################
param_string += '\n# l_max='+str(l_max)+', n_s_mod='+str(n_s_mod)
integ_method = 'lg'
pts_per_osc = 12.*3./2.
## # Need to check evolution_funcs and
## # orthog_mixed_basis if you change this.
beta_activation = (1e-4*3*k_max)**2
beta_margin = 1 #1
if 'dbi' in label:
    N_start_integ   = min(np.log(k_min*c_s_0/H_0)-2.5, 0.0)
else:
    N_start_integ   = 1.0-1.0#3.#1.#2.#3.
num_exps_cs = max(65, 2*l_max)
Nk = 550//2#*2#//2
Nk_mulx = 200
dt_pref = 0.5*0.5*1.*2*np.pi*H_0/(3*k_max*pts_per_osc*c_s_0)
early_del_t = 10*dt_pref*np.exp(N_start_integ*0+2-0.0)
late_del_t  = 5e-4
early_del_t = min(early_del_t,late_del_t*10)
late_t_switch = np.log(np.sqrt(k_min*k_max*c_s_0**2)/H_0)#+1+1
## # Default is 8, set to 8.5 for better DBI at low c_s
## # i.e. beta_dbi_IR = 2.9e-1*2.
delta_early = 8.+0.5
delta_late = -2#+4#+5
param_string += '\n# delta_early = '+str(delta_early)
param_string += '\n# delta_late  = '+str(delta_late)

c_s_atol_scale = (c_s_0*np.exp(-t_init)/H_0)*(7.9e-6/4.6e-3)*(7e-4)
bkgd_atols = np.array([1e-10*c_s_atol_scale]+[abs(phi_0)*1e-6]+[abs(phi_dash_0)*1e-6]+[H_0*1e-6])
bkgd_rtols = np.array([1e-20]*4)
eps_0 = 0.5*phi_dash_0**2/c_s_0
atol_scale = (7e-7)*math.sqrt(c_s_0/eps_0)#*100
zeta_atols = np.array([(1e-12)*atol_scale]*Nk*4)
zeta_rtols = np.array([1e-6]*Nk*4)

if ('tanh' in label) or ('bump' in label) or ('reso' in label):
    delta_late = 1.5
################################################
