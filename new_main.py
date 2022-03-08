import sys
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
import math
from time import time
try:
    from gl_c_wrapper import leggauss_c as leggauss
    ## # Python doesn't catch the failed import until the function is called.
    leggauss(3)
except:
    print('Using Numpy leggauss--- this is less accurate than QUADPTS, so be careful with convergence.')
    from numpy.polynomial.legendre import leggauss
from print_funcs import ps_planck_compar,tidied_coeff_print,tidied_sqz_check,tidied_eql_print,tidied_various_lines_print, print_ps_in_w_ks
from orthog_mixed_basis_3d import tidied_eval,corr,print_tetra,set_up_flat_basis,get_coeffs_1d_from_samples,eval_vander,basis_vander,gen_series_product_matrix,gen_exp_coeffs,tidy_coeffs,decomp_zs,construct_and_integrate,reduce_coeffs_augd
import orthog_mixed_basis_3d as ort
import potential_def_arg
import config
import evolution_funcs

print(config.param_string)
print('# ICs    :',config.phi_0,config.phi_dash_0,config.H_0,config.c_s_0)
print('# k-range:',config.k_exp_min,config.k_exp_max,config.Nk,config.k_min,config.k_max)
print('# dt     :',config.dt_pref,config.early_del_t,config.late_del_t)
print('# ltswit :',config.late_t_switch)
sys.stdout.flush()

t0 = time()
####################################################################
## Get bkgd interps.
t1 = time()
## [dtau_s,eps,eps_s,eta,P,Q,phi,c_s,H,tau_s,N_cross(k)]
bkgd_interps,bkgd_arrays,As,Bs = evolution_funcs.get_bkgd()
t2 = time()
print('# Time background:',t2-t1)
sys.stdout.flush()
####################################################################
## Print bkgd
bkgd_labels = 't, tau_s, phi, dphi, H, V, dV, c_s, eps_s, eta'
np.savetxt('res_bkgd_'+config.label+'.dat',bkgd_arrays[:,::10].T,delimiter=',',header=bkgd_labels,comments='')
####################################################################
## Correlate with Adshead template, if appropriate.
if 'malda' in config.label and 'tanh' in config.label:
    if config.bkgd_label=='malda':
        V_unpert,V_1_unpert,V_11_unpert = potential_def_arg.quadratic_potential_funcs(config.m)
    elif config.bkgd_label=='staro':
        V_unpert,V_1_unpert,V_11_unpert = potential_def_arg.starobinsky_pot_funcs(config.lamb,config.Mpl)
    #ads_ps, ads_bs = ads.gen_ads_template_from_bkgd(bkgd_interps,config.k_min,config.k_max,config.c,config.d,config.phi_f,V_unpert,V_1_unpert,V_11_unpert)
    #np.save("ads_ps.npy",ads_ps)
    #np.save("ads_bs.npy",ads_bs)
####################################################################
## Get zk arrays.
t3 = time()
###############################
kbars,lg_ws = leggauss(config.Nk)
t_zs,t_list = evolution_funcs.get_t_zs()
ks = np.zeros(config.Nk)
###############################
print('# Starting evolution', flush=True)
ks,zs,dzs = evolution_funcs.get_zks(kbars,t_zs,t_list,bkgd_interps,As,Bs,bkgd_arrays[0])
'''split = config.Nk//2
ks[:split],zs[:,:split],dzs[:,:split] = evolution_funcs.get_zks(kbars[:split],t_zs,t_list,bkgd_interps)
ks[split:],zs[:,split:],dzs[:,split:] = evolution_funcs.get_zks(kbars[split:],t_zs,t_list,bkgd_interps)'''
print('# Memory of ks,lg_ws,t_zs,zs,dzs:',ks.nbytes*1e-9,lg_ws.nbytes*1e-9,t_zs.nbytes*1e-9,zs.nbytes*1e-9,dzs.nbytes*1e-9)
t4 = time()
print('# Time modes:',t4-t3)
modes_to_print = [t_zs,zs[:,0].real,zs[:,0].imag,zs[:,config.Nk//2].real,zs[:,config.Nk//2].imag,zs[:,-1].real,zs[:,-1].imag]
modes_to_print = np.array(modes_to_print).T
modes_to_print_labels = 't, z[0]_{real}, z[0]_{imag}, z[N/2]_{real}, z[N/2]_{imag}, z[-1]_{real}, z[-1]_{imag}'
np.savetxt('res_modes_time_evol_'+config.label+'.dat',modes_to_print[::100],delimiter=',', header=modes_to_print_labels,comments='')
print('# len t_zs',len(t_zs))
t4 = time()
print('# Time modes:',t4-t3)
print('# Memory of zs, dzs:',zs.nbytes*1e-9,dzs.nbytes*1e-9)
sys.stdout.flush()
####################################################################
## Print PS, ns, save interps.
final_zetas = zs[-1]*np.exp(-1j*ks*bkgd_interps[9](t_zs[-1]))
R2 = (np.conjugate(final_zetas)*final_zetas).real
logR2s = np.log(R2)
ps_eval = InterpolatedUnivariateSpline(ks,logR2s)
np.save("res_logR2_interp.npy",ps_eval)
dps_eval = ps_eval.derivative()
f_name = 'res_ps_planck_'+config.label+'.csv'
## # [dtau_s,eps,eps_s,eta,P,Q,phi,c_s,H,tau_s,N_cross(k)]
print_ps_in_w_ks(f_name, config.k_min, config.k_max, config.k_pivot, ps_eval, dps_eval, bkgd_interps[8], bkgd_interps[1], bkgd_interps[7], bkgd_interps[3], bkgd_interps[2], bkgd_interps[10])
if not config.calc_bis:
    sys.exit()
####################################################################
## Choose lmax, set up basis.
l_max = config.l_max
if not config.logbasis:
    working_basis = ort.set_up_mixed_basis(*config.working_basis_params)
    final_basis   = ort.set_up_mixed_basis(*config.final_basis_params)
else:
    working_basis = ort.set_up_log_basis(config.k_min, config.k_max, l_max, inv=False, verbose=True)
    final_basis   = ort.set_up_log_basis(config.k_min, config.k_max, l_max+3, inv=True, verbose=True)
basis_funcs = np.copy(working_basis)
print('# n_s_mod:',config.n_s_mod)
print('# Ortho check:', np.sum(ort.check_orthog(basis_funcs,config.k_min,config.k_max)), flush=True)
####################################################################
## Get Ip, Jp.
t5 = time()
I_coeffs_final,J_coeffs_final,fit_checks = decomp_zs(ks,lg_ws,t_zs,zs,dzs,bkgd_interps,basis_funcs,l_max)
del zs
del dzs
t6 = time()
print('# Time decomp:',t6-t5)
#np.savetxt('res_fit_check_'+config.label+'_l'+str(l_max)+'.dat',fit_checks,delimiter=',',comments='')
####################################################################
## Integrate a_pqr's
t7 = time()
coeffs, basis_funcs_padded, integrands_to_print, to_print_labels, raw_coeffs = construct_and_integrate(t_zs,I_coeffs_final,J_coeffs_final,bkgd_interps,basis_funcs,l_max, final_basis)
#np.savetxt('res_integrand_check_'+config.label+'_l'+str(l_max)+'.dat',integrands_to_print,delimiter=',',header=to_print_labels,comments='')
t8 = time()
print('# Time construct and integrate:',t8-t7)

sys.stdout.flush()
####################################################################
l_max = len(basis_funcs_padded)
coeffs = coeffs[:l_max, :l_max, :l_max]
f_name = 'res_coeffs_'+config.label+'_l'+str(l_max)+'.csv'
tidied_coeff_print(config.k_min, config.k_max, coeffs, basis_funcs_padded, f_name, config.final_basis_params)
f_name = 'res_sqz_'+config.label+'_l'+str(l_max)+'.csv'
k_mark = 0.5*(config.k_min+config.k_max)
tidied_sqz_check(config.k_min, config.k_max, k_mark, coeffs, basis_funcs_padded, ps_eval, f_name)
####################################################################
f_name = 'res_eql_'+config.label+'_l'+str(l_max)+'.csv'
tidied_eql_print(config.k_min,config.k_max,coeffs,basis_funcs_padded,f_name,compare_func=0)
f_name = 'res_various_'+config.label+'_l'+str(l_max)+'.csv'
tidied_various_lines_print(config.k_min,config.k_max,coeffs,basis_funcs_padded,f_name,ps_eval)
####################################################################
print('# Time total:',time()-t0)

