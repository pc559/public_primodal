'''
This is a mix of functions used to set up basis sets, decompose functions
in those basis sets, construct and perform integrals.
A lot of the functions here have tidier versions in general_coeff_funcs.py,
which also has some comments.
'''

import numpy as np
import math
try:
    from gl_c_wrapper import leggauss_c as leggauss
    ## # Python doesn't catch the failed import until the function is called.
    leggauss(3)
except:
    print('Using Numpy leggauss--- this is less accurate than QUADPTS, so be careful with convergence.')
    from numpy.polynomial.legendre import leggauss
from numpy.polynomial.legendre import legvander
from scipy.special import eval_legendre,jv,gamma,expi,sici
from scipy.integrate import cumtrapz
from time import time
import sys
import config
from arb_pts_integrator import simps_osc_weights_lg
import quad_sum

## Move to config?
LG_INV_FUNC_RES = 600
LG_LOW_RES = 75
Nk_aug = 2500

def tidied_coeff_print(k_min,k_max,coeffs,basis_funcs,f_name):
    data_file = open(f_name,'w')
    l = len(coeffs[0,0,:])
    data_file.write( '"i","j","k","a_ijk"#'+str(k_min)+','+str(k_max)+'\n' )
    for i in range(l):
        for j in range(l):
            for k in range(l):
                data_file.write( str(i)+','+str(j)+','+str(k)+','+str(coeffs[i,j,k])+'\n' )
            data_file.write( '\n' )
        data_file.write( '\n' )
    data_file.close()

def load_coeffs(filename):
    with open(filename) as f:
            first_line = f.readline().strip().split('#')[1]

    k_min,k_max = np.array(first_line.split(',')[0:2],dtype=np.float64)

    coeffs = np.loadtxt(filename,delimiter=',',skiprows=1)
    L3 = np.shape(coeffs)[0]
    L = int(round(L3**(1./3)))
    coeffs = coeffs.reshape((L,L,L,4))
    coeffs = coeffs[:,:,:,3]
    return k_min,k_max,coeffs

def in_tetra(p,lower_lim=0):
    x,y,z = p
    return ( (x+y>=z) and (y+z>=x) and (z+x>=y) and (x+y+z>lower_lim) )

def legmulx_no_trim(c): # as in numpy legendre module, but without trimming zeros.
    prd = np.empty(len(c) + 1, dtype=c.dtype)
    prd[0] = c[0]*0
    prd[1] = c[0]
    for i in range(1, len(c)):
        j = i + 1
        k = i - 1
        s = i + j
        prd[j] = (c[i]*j)/s
        prd[k] += (c[i]*i)/s
    return prd

def bar(x,k_min,k_max):
    return (2*x-(k_min+k_max))/(k_max-k_min)

def unbar(x,k_min,k_max):
    return 0.5*((k_max-k_min)*x+k_min+k_max)

def gen_vs(k_min,k_max,Nb,Nk=LG_INV_FUNC_RES):
    xs,ws = leggauss(Nk)
    xs_ub = unbar(xs,k_min,k_max)
    Is = 1.0/xs_ub
    lV = legvander(xs,Nb-2)
    vs = np.dot(lV.T,Is*ws)*np.sqrt((2.*np.arange(np.shape(lV)[1])+1.)/2.)
    return vs

def mixed_basis_eval_3d(k_min,k_max,xs,ys,zs,coeffs,basis_funcs):
    xs,ys,zs = bar(np.array([xs,ys,zs]),k_min,k_max)
    res = np.zeros_like(xs)
    f1s = np.array([ f(xs) for f in basis_funcs ])
    f2s = np.array([ f(ys) for f in basis_funcs ])
    f3s = np.array([ f(zs) for f in basis_funcs ])
    res = np.einsum( 'ijk,iq,jq,kq->q', coeffs,f1s,f2s,f3s, optimize=True)
    return res

def zero_func(x):
    return np.zeros_like(x)

def basis_vander(basis_funcs,k_min,k_max,decomp_xs,decomp_ws):
    norms = np.zeros(len(basis_funcs))
    for i,b in enumerate(basis_funcs):
        bs = b(decomp_xs)
        norms[i] = np.dot(bs**2,decomp_ws)

    vander = np.zeros((len(basis_funcs),len(decomp_xs)))
    for i,x in enumerate(decomp_xs):
        for j,b_func in enumerate(basis_funcs):
            vander[j,i] = b_func(x)*decomp_ws[i]/norms[j]
    return vander

def eval_vander(basis_funcs,k_min,k_max,decomp_xs):
    vander = np.zeros((len(basis_funcs),len(decomp_xs)))
    for i,x in enumerate(decomp_xs):
        for j,b_func in enumerate(basis_funcs):
            vander[j,i] = b_func(x)
    return vander

def get_coeffs_1d_from_samples(fs,vander):
    #coeffs = np.einsum('ij,j->i',vander,fs)
    coeffs = np.dot(vander,fs)
    return coeffs

def get_coeffs_1d(func_to_fit,basis_funcs,k_min,k_max,Nk=Nk_aug):
    #print "Warning: low sampling rate"
    decomp_xs,decomp_ws = leggauss(Nk)
    norms = np.zeros(len(basis_funcs))
    for i,b in enumerate(basis_funcs):
        fs = b(decomp_xs)
        norms[i] = np.dot(fs**2,decomp_ws)

    vander = np.zeros((len(basis_funcs),len(decomp_xs)))
    for j,b_func in enumerate(basis_funcs):
        vander[j,:] = b_func(decomp_xs)*decomp_ws[:]/norms[j]
    
    f_evals = np.zeros(Nk)
    for i,x in enumerate(decomp_xs):
        ux = unbar(x,k_min,k_max)
        f_evals[i] = func_to_fit(ux)

    coeffs = np.dot(vander,f_evals)
    return coeffs

def corr(k_min,k_max,func_to_fit,coeffs,basis_funcs,Nk=LG_LOW_RES,ampl=False):
    decomp_xs,decomp_ws = leggauss(Nk)

    vander = np.zeros((len(basis_funcs),len(decomp_xs)))
    for j,b_func in enumerate(basis_funcs):
        vander[j,:] = b_func(decomp_xs)
    
    Nk = len(decomp_xs)
    f_evals = np.zeros((Nk,Nk,Nk))
    weights = np.zeros((Nk,Nk,Nk))
    for i,x in enumerate(decomp_xs):
        for j,y in enumerate(decomp_xs):
            ux,uy = unbar(np.array([x,y]),k_min,k_max)
            uzs = unbar(decomp_xs,k_min,k_max)
            in_tetra = (ux<=uy+uzs)*(uy<=uzs+ux)*(uzs<=ux+uy)
            f_evals[i,j,:] = func_to_fit(ux,uy,uzs)*in_tetra
            weights[i,j,:] = decomp_ws[i]*decomp_ws[j]*decomp_ws*in_tetra

    basis_evals = np.einsum('ji,jqr->iqr',vander,coeffs, optimize=True)
    basis_evals = np.einsum('ji,qjr->qir',vander,basis_evals, optimize=True)
    basis_evals = np.einsum('ji,qrj->qri',vander,basis_evals, optimize=True)

    fg = np.einsum('ijk,ijk->',basis_evals,f_evals*weights, optimize=True)
    ff = np.einsum('ijk,ijk->',f_evals,f_evals*weights, optimize=True)
    gg = np.einsum('ijk,ijk->',basis_evals,basis_evals*weights, optimize=True)

    if ampl:
        return fg,ff,gg,fg/np.sqrt(ff*gg),np.sqrt(1+(ff-2*fg)/gg)
    else:
        return fg/np.sqrt(ff*gg)

def check_conv(k_min,k_max,coeffs,base_basis,aug_funcs,p_maxs,Nk=LG_LOW_RES,cube=False,full_evals=None,return_evals=False):
    coeffs = np.copy(coeffs)
    basis_funcs = np.copy(base_basis)
    for af in aug_funcs:
        basis_funcs = augment_basis_arb_func(k_min,k_max,basis_funcs,af)
    decomp_xs,decomp_ws = leggauss(Nk)

    vander = np.zeros((len(basis_funcs),len(decomp_xs)))
    for i,x in enumerate(decomp_xs):
        for j,b_func in enumerate(basis_funcs):
            vander[j,i] = b_func(x)
    
    Nk = len(decomp_xs)
    weights = np.zeros((Nk,Nk,Nk))
    for i,x in enumerate(decomp_xs):
        for j,y in enumerate(decomp_xs):
            for k,z in enumerate(decomp_xs):
                ux,uy,uz = unbar(np.array([x,y,z]),k_min,k_max)
                if (ux<=uy+uz and uy<=uz+ux and uz<=ux+uy) or cube:# and ux+uy+uz<2*k_max:
                    weights[i,j,k] = decomp_ws[i]*decomp_ws[j]*decomp_ws[k]

    if full_evals is None:
        full_basis_evals = np.einsum('ji,jqr->iqr',vander,coeffs, optimize=True)
        full_basis_evals = np.einsum('ji,qjr->qir',vander,full_basis_evals, optimize=True)
        full_basis_evals = np.einsum('ji,qrj->qri',vander,full_basis_evals, optimize=True)
    else:
        full_basis_evals = np.copy(full_evals)
    full_basis_norm = np.einsum('ijk,ijk->',full_basis_evals,full_basis_evals*weights, optimize=True)

    corrs = []
    for p in p_maxs:
        if p>=3:
            new_coeffs,new_basis_funcs = reduce_coeffs_augd(k_min,k_max,coeffs,base_basis,aug_funcs,p)
            for j,b_func in enumerate(new_basis_funcs[:len(aug_funcs)]):
                vander[j,:] = b_func(decomp_xs)
            new_basis_evals = np.einsum('ji,jqr->iqr',vander[:p],new_coeffs, optimize=True)
            new_basis_evals = np.einsum('ji,qjr->qir',vander[:p],new_basis_evals, optimize=True)
            new_basis_evals = np.einsum('ji,qrj->qri',vander[:p],new_basis_evals, optimize=True)
            fg = np.einsum('ijk,ijk->',new_basis_evals,full_basis_evals*weights, optimize=True)
            ff = np.einsum('ijk,ijk->',new_basis_evals,new_basis_evals*weights, optimize=True)
            print(cube,np.sqrt(ff),np.sqrt(fg),np.sqrt(full_basis_norm))
            corrs.append(np.sqrt(1+(ff-2*fg)/full_basis_norm))
        else:
            corrs.append(0)

    if return_evals:
        return corrs,full_basis_evals
    else:
        return corrs

def fnl_with_templates(k_min,k_max,templates,coeffs,basis_funcs,A):
    decomp_xs,decomp_ws = leggauss(LG_LOW_RES)

    vander = np.zeros((len(basis_funcs),len(decomp_xs)))
    for i,x in enumerate(decomp_xs):
        for j,b_func in enumerate(basis_funcs):
            vander[j,i] = b_func(x)
    
    Nk = len(decomp_xs)
    f_evals = np.zeros((len(templates),Nk,Nk,Nk))
    weights = np.zeros((Nk,Nk,Nk))
    for i,x in enumerate(decomp_xs):
        for j,y in enumerate(decomp_xs):
            for k,z in enumerate(decomp_xs):
                ux,uy,uz = unbar(np.array([x,y,z]),k_min,k_max)
                if ux<=uy+uz and uy<=uz+ux and uz<=ux+uy:# and ux+uy+uz<2*k_max:
                    weights[i,j,k] = decomp_ws[i]*decomp_ws[j]*decomp_ws[k]
                    for p,templ in enumerate(templates):
                        f_evals[p,i,j,k] = templ(ux,uy,uz)

    basis_evals = np.einsum('ji,jqr->iqr',vander,coeffs, optimize=True)
    basis_evals = np.einsum('ji,qjr->qir',vander,basis_evals, optimize=True)
    basis_evals = np.einsum('ji,qrj->qri',vander,basis_evals, optimize=True)

    template_norms = np.zeros_like(templates)
    cross_corrs = np.zeros_like(templates)
    for p in range(len(templates)):
        template_norms[p] = np.einsum('ijk,ijk->',f_evals[p],f_evals[p]*weights, optimize=True)
        cross_corrs[p] = np.einsum('ijk,ijk->',basis_evals,f_evals[p]*weights, optimize=True)

    gg = np.einsum('ijk,ijk->',basis_evals,basis_evals*weights, optimize=True)
    return (cross_corrs/template_norms)*(1./(6*(A*2*np.pi**2)**2))*5./3.

def eval_on_grid(k_min,k_max,coeffs,basis_funcs,points,zero_non_tetra=False, cube=False):
    bpoints = bar(points,k_min,k_max)
    vander = np.zeros((len(basis_funcs),len(points)))
    for j,b_func in enumerate(basis_funcs):
        vander[j,:] = b_func(bpoints)
    
    basis_evals = np.einsum('ji,jqr->iqr', vander, coeffs, optimize=True)
    basis_evals = np.einsum('ji,qjr->qir', vander, basis_evals, optimize=True)
    basis_evals = np.einsum('ji,qrj->qri', vander, basis_evals, optimize=True)
    
    Np = len(points)
    m = np.meshgrid(points,points,points)
    xs = m[0].reshape(Np**3)
    ys = m[1].reshape(Np**3)
    zs = m[2].reshape(Np**3)
    triangle_ineq = (xs<=ys+zs)*(ys<=zs+xs)*(zs<=xs+ys)#*(xs+ys+zs<2*xs[-1])
    points_in_tetra = (triangle_ineq).reshape((Np,Np,Np))
    if not cube:
        basis_evals *= points_in_tetra
    if not zero_non_tetra:
        xs = xs[triangle_ineq]
        ys = ys[triangle_ineq]
        zs = zs[triangle_ineq]
        basis_evals = basis_evals[basis_evals!=0]
    return xs,ys,zs,basis_evals

def get_conversion_matrix(old_basis,new_basis,k_min,k_max):
    Q = np.zeros((len(new_basis),len(old_basis)))
    for i,ob in enumerate(old_basis):
        Q[:,i] = get_coeffs_1d(ob,new_basis,-1,1)
    return Q

def convert_between_bases(old_basis,new_basis,k_min,k_max,coeffs):
    Q = get_conversion_matrix(old_basis,new_basis,k_min,k_max)
    coeffs = np.copy(coeffs)
    coeffs = np.einsum('ij,jqr->iqr', Q, coeffs, optimize=True)
    coeffs = np.einsum('ij,qjr->qir', Q, coeffs, optimize=True)
    coeffs = np.einsum('ij,qrj->qri', Q, coeffs, optimize=True)
    return coeffs

def reduce_coeffs_augd(k_min,k_max,coeffs,base_basis,aug_funcs,new_Nb):
    new_basis = np.copy(base_basis[:new_Nb-len(aug_funcs)])
    for af in aug_funcs:
        new_basis = augment_basis_arb_func(k_min,k_max,new_basis,af)
    old_basis = np.copy(base_basis)
    for af in aug_funcs:
        old_basis = augment_basis_arb_func(k_min,k_max,old_basis,af)
    Q = get_conversion_matrix(old_basis,new_basis,k_min,k_max)

    coeffs = np.copy(coeffs)
    coeffs = np.einsum('ij,jqr->iqr',Q,coeffs, optimize=True)
    coeffs = np.einsum('ij,qjr->qir',Q,coeffs, optimize=True)
    coeffs = np.einsum('ij,qrj->qri',Q,coeffs, optimize=True)
    return coeffs,new_basis

def coeffs_mulx_mixed(coeffs,k_min,k_max,vs,pad=False):
    xcoeffs = np.zeros(len(coeffs)+1*pad)
    xcoeffs[0] = 0.
    a,b = (k_min+k_max)*0.5,(k_max-k_min)*0.5
    xcoeffs[1:len(vs)+1]  = a*(coeffs[1:]-coeffs[0]*vs)
    xcoeffs[1:]         += b*(legmulx_no_trim(coeffs[1:]-coeffs[0]*vs))[:len(vs)+1*pad]
    xcoeffs[1] += coeffs[0]
    return xcoeffs

def coeffs_mulx_flat(coeffs,k_min,k_max,vs,pad=False):
    xcoeffs = np.zeros(len(coeffs))
    a,b = (k_min+k_max)*0.5,(k_max-k_min)*0.5
    xcoeffs = a*coeffs
    xcoeffs += b*(legmulx_no_trim(coeffs))[:len(xcoeffs)]
    return xcoeffs

def gen_mulx_matrix(basis_in, basis_out, k_min, k_max, power=1, Nk=Nk_aug):
    Nb_in = len(basis_in)
    Nb_out = len(basis_out)
    mul_matrix = np.zeros((Nb_out, Nb_in))
    for i in range(Nb_in):
        mul_matrix[:, i] = get_coeffs_1d(lambda k: basis_in[i](bar(k, k_min, k_max))*k**power, basis_out, k_min, k_max, Nk=Nk)
    return mul_matrix.T

def check_decomp_1d(basis_funcs, k_min, k_max, func, cs, Nk=Nk_aug):
    xs,ws = leggauss(Nk)
    Nb = len(basis_funcs)
    if Nb>70:
        print("Warning: low sampling rate")
        return
    vander = np.array([b(xs) for b in basis_funcs])
    recomp = np.einsum('ji,j->i', vander, cs, optimize=True)
    evals = func(unbar(xs,k_min,k_max))
    diffs = evals-recomp
    return np.sqrt(np.sum(diffs**2)/np.sum(evals**2))

def gen_series_product_matrix_old(basis_funcs,k_min,k_max,ext=-1):
    Nb = len(basis_funcs)
    if ext==-1:
        second_set = basis_funcs
    else:
        second_set = [zero_func]*(Nb+ext)
        second_set[:Nb] = basis_funcs
        for i in range(Nb,Nb+ext):
            second_set[i] = lambda x,i=i:eval_legendre(i-1,x)*math.sqrt((2*i-1)/2.)
    decomp_xs,decomp_ws = leggauss(LG_INV_FUNC_RES)
    norms = np.zeros(len(basis_funcs))
    for i,b in enumerate(basis_funcs):
        fs = b(decomp_xs)
        norms[i] = np.sqrt(np.dot(fs**2,decomp_ws))

    vander = np.zeros((len(basis_funcs),len(decomp_xs)))
    for i,x in enumerate(decomp_xs):
        for j,b_func in enumerate(basis_funcs):
            vander[j,i] = b_func(x)*decomp_ws[i]/norms[j]

    integrand = np.zeros((len(second_set),len(basis_funcs),len(decomp_xs)))
    for i,x in enumerate(decomp_xs):
        for j,b_func1 in enumerate(basis_funcs):
            for k,b_func2 in enumerate(second_set):
                integrand[k,j,i] = b_func1(x)*b_func2(x)
    
    product_matrix = np.einsum('pq,ijq->ijp',vander,integrand, optimize=True)
    return product_matrix

def gen_series_product_matrix(in_basisA,in_basisB,out_basis,k_min,k_max):
    decomp_xs,decomp_ws = leggauss(LG_INV_FUNC_RES)

    vander = np.zeros((len(out_basis),len(decomp_xs)))
    for i,x in enumerate(decomp_xs):
        for j,b_func in enumerate(out_basis):
            vander[j,i] = b_func(x)*decomp_ws[i]

    integrand = np.zeros((len(in_basisA),len(in_basisB),len(decomp_xs)))
    for i,x in enumerate(decomp_xs):
        for j,b_func1 in enumerate(in_basisA):
            for k,b_func2 in enumerate(in_basisB):
                integrand[j,k,i] = b_func1(x)*b_func2(x)
    
    product_matrix = np.einsum('pq,ijq->ijp', vander, integrand, optimize=True)
    return product_matrix

def gen_exp_coeffs(k_min,k_max,tau_s,p_max,norm_0=1,vs=0):
    ## # ASSUMES FLAT LEGENDRE BASIS
    ## # Should prob change this to use simps_osc
    ## # to calc the coeffs directly,
    ## # indep of the basis.
    tau_s = np.repeat(tau_s[:,np.newaxis], p_max, axis=1)
    t = -(k_max-k_min)*tau_s*0.5
    cs = np.zeros((np.shape(tau_s)[0],p_max))*1j
    rs = np.arange(p_max)
    pref = 0.5*(2*rs+1)*np.exp(0.5j*(k_min+k_max)*tau_s)*(-1j)**rs
    small_t_inds = np.where(np.abs(t[:,0])<=1e-7)[0]
    large_t_inds = np.where(np.abs(t[:,0])>1e-7)[0]
    for i,r in enumerate(rs):
        pref[small_t_inds,i] *= np.sqrt(np.pi)*(((0.5*t[small_t_inds,i])**r)/gamma(r+1.5)-((0.5*t[small_t_inds,i])**(r+2))/gamma(r+2.5))*np.sqrt(2./(1+2*i))
        pref[large_t_inds,i] *= np.sqrt(2*np.pi/t[large_t_inds,i])*jv(r+0.5,t[large_t_inds,i])*np.sqrt(2./(1+2*i))
    cs[:,0:] = pref
    return cs

def coeffs_algebra_3d(coeffs, basis_funcs, k_min, k_max, mulp, mulq, mulr):
    Nb = len(coeffs[0, 0, :])
    temp_coeffs = np.copy(coeffs)
    if mulp is not None:
        temp_coeffs = np.einsum('ip,ijk->pjk', mulp, temp_coeffs, optimize=True)
    if mulq is not None:
        temp_coeffs = np.einsum('jp,ijk->ipk', mulq, temp_coeffs, optimize=True)
    if mulr is not None:
        temp_coeffs = np.einsum('kp,ijk->ijp', mulr, temp_coeffs, optimize=True)
    return temp_coeffs

def tidy_coeffs(coeffs, basis_funcs, k_min, k_max, basis_funcs_padded):
    Nb = len(coeffs[0,0,0,:])
    cs_diff = len(basis_funcs_padded) - len(basis_funcs)
    new_coeffs = np.zeros((6,Nb+cs_diff,Nb+cs_diff,Nb+cs_diff))
    # Derivative -> -ik
    # Dealing with k factors:
    # z is the different one.
    # Sym in x, y, z.
    # Every zeta gets an inverse factor.
    tA = time()
    Nk_mulx = config.Nk_mulx
    div2_matrix = gen_mulx_matrix(basis_funcs, basis_funcs_padded, k_min, k_max, power=-2, Nk=Nk_mulx)
    div1_matrix = gen_mulx_matrix(basis_funcs, basis_funcs_padded, k_min, k_max, power=-1, Nk=Nk_mulx)
    id_matrix = gen_mulx_matrix(basis_funcs, basis_funcs_padded, k_min, k_max, power=0, Nk=Nk_mulx)
    mul1_matrix = gen_mulx_matrix(basis_funcs, basis_funcs_padded, k_min, k_max, power=1, Nk=Nk_mulx)
    mul2_matrix = gen_mulx_matrix(basis_funcs, basis_funcs_padded, k_min, k_max, power=2, Nk=Nk_mulx)
    mul3_matrix = gen_mulx_matrix(basis_funcs, basis_funcs_padded, k_min, k_max, power=3, Nk=Nk_mulx)
    ################################################################################
    ## # coeffs_algebra_3d(coeffs, basis_funcs, k_min, k_max, mulp, mulq, mulr) # ##
    ################################################################################
    tB = time()
    ## # Including gradients, symming in k1 and k2, including 1/k3 factor for each zeta (not deriv).
    if 0 in config.shape_indices:
        ind = np.where(config.shape_indices==0)[0][0]
        new_coeffs[0] = 2*coeffs_algebra_3d(coeffs[ind],basis_funcs_padded,k_min,k_max,id_matrix,id_matrix,id_matrix)
    if 1 in config.shape_indices:
        ind = np.where(config.shape_indices==1)[0][0]
        new_coeffs[1] = 2*coeffs_algebra_3d(coeffs[ind],basis_funcs_padded,k_min,k_max,id_matrix,id_matrix,div1_matrix)
    if 2 in config.shape_indices:
        ind = np.where(config.shape_indices==2)[0][0]
        new_coeffs[2] = -coeffs_algebra_3d(coeffs[ind],basis_funcs_padded,k_min,k_max,div1_matrix,div1_matrix,mul1_matrix)
        new_coeffs[2] += coeffs_algebra_3d(coeffs[ind],basis_funcs_padded,k_min,k_max,div1_matrix,mul1_matrix,div1_matrix)
        new_coeffs[2] += coeffs_algebra_3d(coeffs[ind],basis_funcs_padded,k_min,k_max,mul1_matrix,div1_matrix,div1_matrix)
    if 3 in config.shape_indices:
        ind = np.where(config.shape_indices==3)[0][0]
        new_coeffs[3] = -coeffs_algebra_3d(coeffs[ind],basis_funcs_padded,k_min,k_max,id_matrix,id_matrix,div1_matrix)
        new_coeffs[3] += 0.5*coeffs_algebra_3d(coeffs[ind],basis_funcs_padded,k_min,k_max,mul2_matrix,div2_matrix,div1_matrix)
        new_coeffs[3] -= 0.5*coeffs_algebra_3d(coeffs[ind],basis_funcs_padded,k_min,k_max,id_matrix,div2_matrix,mul1_matrix)
        new_coeffs[3] += 0.5*coeffs_algebra_3d(coeffs[ind],basis_funcs_padded,k_min,k_max,div2_matrix,mul2_matrix,div1_matrix)
        new_coeffs[3] -= 0.5*coeffs_algebra_3d(coeffs[ind],basis_funcs_padded,k_min,k_max,div2_matrix,id_matrix,mul1_matrix)
    if 4 in config.shape_indices:
        ind = np.where(config.shape_indices==4)[0][0]
        new_coeffs[4] = coeffs_algebra_3d(coeffs[ind],basis_funcs_padded,k_min,k_max,div2_matrix,div2_matrix,mul3_matrix)
        new_coeffs[4] -= coeffs_algebra_3d(coeffs[ind],basis_funcs_padded,k_min,k_max,id_matrix,div2_matrix,mul1_matrix)
        new_coeffs[4] -= coeffs_algebra_3d(coeffs[ind],basis_funcs_padded,k_min,k_max,div2_matrix,id_matrix,mul1_matrix)
    if 5 in config.shape_indices:
        ## # eg (2.22) of astro-ph/0507053
        ind = np.where(config.shape_indices==5)[0][0]
        new_coeffs[5] = coeffs_algebra_3d(coeffs[ind],basis_funcs_padded,k_min,k_max,mul1_matrix,div1_matrix,id_matrix)
        new_coeffs[5] += coeffs_algebra_3d(coeffs[ind],basis_funcs_padded,k_min,k_max,div1_matrix,mul1_matrix,id_matrix)
        new_coeffs[5] -= coeffs_algebra_3d(coeffs[ind],basis_funcs_padded,k_min,k_max,div1_matrix,div1_matrix,mul2_matrix)
    new_coeffs *= 2.
    sum_coeffs = new_coeffs.sum(axis=0)
    tC = time()
    for i in range(6):
        if i in config.shape_indices:
            f_name = 'res_tidied_coeffs_v'+str(i)+'_l'+str(len(basis_funcs_padded))+'.dat'
    tD = time()
    print('# Full:', sum_coeffs.sum())
    print('# Sums:', [(config.shape_indices[i],new_coeffs[i].sum()) for i in range(len(config.shape_indices))])
    print('# L1s:', [(config.shape_indices[i],(np.abs(new_coeffs[i])).sum()) for i in range(len(config.shape_indices))])
    print('# L2s:', [(config.shape_indices[i],np.sqrt((new_coeffs[i]**2).sum())) for i in range(len(config.shape_indices))])
    final_coeffs = np.copy(np.einsum('ijk->ijk',sum_coeffs))
    final_coeffs += np.copy(np.einsum('ijk->jki',sum_coeffs))
    final_coeffs += np.copy(np.einsum('ijk->kij',sum_coeffs))
    tE = time()
    print('# Time tidy:',tE-tD,tD-tC,tC-tB,tB-tA)
    return final_coeffs, basis_funcs_padded

def tidied_eval(k_min,k_max,coeffs,basis_funcs,points):
    l = len(points[0])
    res = np.zeros(l)
    if l==0:
        return res
    xs = points[0,:]
    ys = points[1,:]
    zs = points[2,:]
    if l<=10**4:
        res = mixed_basis_eval_3d(k_min,k_max,xs,ys,zs,coeffs,basis_funcs)
    else:
        temp_a = mixed_basis_eval_3d(k_min,k_max,xs[:10**4],ys[:10**4],zs[:10**4],coeffs,basis_funcs)
        temp_b = mixed_basis_eval_3d(k_min,k_max,xs[10**4:],ys[10**4:],zs[10**4:],coeffs,basis_funcs)
        res = np.concatenate([temp_a,temp_b])
    return res

def plot_eql(k_min,k_max,coeffs,basis_funcs,Nk=100,label='',plot=True):
    xs = np.linspace(k_min,k_max,Nk)
    ys = np.linspace(k_min,k_max,Nk)
    zs = np.linspace(k_min,k_max,Nk)
    points = np.array([xs,ys,zs])
    res = tidied_eval(k_min,k_max,coeffs,basis_funcs,points)
    if not plot:
        return xs, res
    else:
        import matplotlib.pyplot as plt
        plt.plot(xs,res,label=label+str(len(basis_funcs)))
        plt.legend()
        plt.show()

def plot_sqz(k_min,k_max,coeffs,basis_funcs,Nk=100,label='', plot=True):
    K = (k_min+k_max)*0.5
    sqz = np.linspace(k_min/K,1,Nk)
    xs = K*(1-0.5*sqz)
    ys = K*(1-0.5*sqz)
    zs = K*(sqz)
    triangle_ineq = (xs<=ys+zs)*(ys<=zs+xs)*(zs<=xs+ys)
    points = np.array([xs,ys,zs])*triangle_ineq
    res = tidied_eval(k_min,k_max,coeffs,basis_funcs,points)#*zs/xs
    if not plot:
        return xs, res
    else:
        import matplotlib.pyplot as plt
        plt.plot(zs/xs,res,label=label+str(len(basis_funcs)))
        plt.legend()
        plt.show()

def print_tetra(filename,k_min,k_max,coeffs,basis_funcs,Nk,ret=False):
    points = np.linspace(k_min,k_max,Nk)
    xs,ys,zs,res = eval_on_grid(k_min,k_max,coeffs,basis_funcs,points)
    to_print = np.array([xs,ys,zs,res]).T
    np.savetxt(filename,to_print,delimiter=',',header='"X","Y","Z","f(X,Y,Z)"',comments='')
    if ret:
        return xs,ys,zs,res

def print_slice(filename,k_min,k_max,coeffs,basis_funcs,Nk):
    points = np.linspace(k_min,k_max,Nk)
    xs,ys,zs,res = eval_on_grid(k_min,k_max,coeffs,basis_funcs,points)
    slice_indices = np.where(xs==ys)
    to_print = np.array([xs[slice_indices],ys[slice_indices],zs[slice_indices],res[slice_indices]]).T
    np.savetxt(filename,to_print,delimiter=',',header='"X","Y","Z","f(X,Y,Z)"',comments='')

def check_orthog(basis_funcs,k_min,k_max):
    test_xs,test_ws = leggauss(LG_INV_FUNC_RES)
    vander = np.array([b(test_xs) for b in basis_funcs])
    p_max = len(basis_funcs)
    res = np.zeros((p_max,p_max))
    for i in range(p_max):
        for j in range(p_max):
            prod = np.sum(test_ws*vander[i,:]*vander[j,:])
            res[i,j] = prod
    return res

def set_up_fourier_basis(k_min, k_max, Nb, Nk=Nk_aug):
    Nb_set = Nb+Nb%2-1
    basis_funcs = [zero_func]*Nb
    normed_basis_funcs = [zero_func]*Nb
    norms = np.zeros(Nb)
    xs, ws = leggauss(Nk)
    basis_funcs[0] = lambda x: np.ones_like(x)
    for r in range(1, (Nb_set-1)//2+1):
        basis_funcs[2*r-1] = lambda x, r=r: np.sin(np.pi*r*x)
        basis_funcs[2*r]   = lambda x, r=r: np.cos(np.pi*r*x)
    if Nb%2==0:
        r = (Nb_set-1)//2+1
        basis_funcs[2*r-1] = lambda x, r=r: np.sin(np.pi*r*x)
    for i in range(Nb):
        fs = basis_funcs[i](xs)
        norms[i] = np.sqrt(np.dot(fs**2, ws))
    for i in range(len(basis_funcs)):
        normed_basis_funcs[i] = lambda x, i=i: basis_funcs[i](x)/norms[i]
    return normed_basis_funcs

def set_up_flat_basis(k_min, k_max, Nb ,Nk=Nk_aug):
    basis_funcs = [zero_func]*Nb
    norms = np.zeros(Nb)
    xs, ws = leggauss(Nk)
    for i in range(Nb):
        fs = eval_legendre(i,xs)
        norms[i] = np.sqrt(np.dot(fs**2, ws))
    for i in range(len(basis_funcs)):
        basis_funcs[i] = lambda x, i=i: eval_legendre(i, x)/norms[i]
    return basis_funcs

def decomp_zs(ks, lg_ws, t_zs, zs, dzs, bkgd_interps, basis_funcs, l_max):
    t0 = time()
    final_zetas = zs[-1]*np.exp(-1j*ks*bkgd_interps[9](t_zs[-1]))
    ## # Memory problems!
    for i,f in enumerate(zs):
        zs[i] = zs[i].conj()
    #zs[:] = zs[:].conj()
    zs *= final_zetas
    for i,df in enumerate(dzs):
        dzs[i] = dzs[i].conj()
    #dzs[:] = dzs[:].conj()
    dzs *= final_zetas
    t1 = time()
    print('# Time conj:',t1-t0)
    sys.stdout.flush()

    kbars = bar(ks,config.k_min,config.k_max)
    I_coeffs = np.zeros((len(t_zs),l_max))*1j
    J_coeffs = np.zeros((len(t_zs),l_max))*1j
    fit_check = np.zeros(len(t_zs)//10+1)*1j
    dfit_check = np.zeros(len(t_zs)//10+1)*1j

    mixed_vander = basis_vander(basis_funcs,config.k_min,config.k_max,kbars,lg_ws)
    test_vander = eval_vander(basis_funcs,config.k_min,config.k_max,kbars)
    tau_s_array = bkgd_interps[9](t_zs)
    t2 = time()
    for i,tau_s in enumerate(tau_s_array):
        if config.late_t_switch<t_zs[i]:
            zs[i] *= np.exp(1j*ks*tau_s)
            dzs[i] *= np.exp(1j*ks*tau_s)
        else:
            i_switch = i

        if config.integ_method=='lg':
            I_coeffs[i] = get_coeffs_1d_from_samples(zs[i],mixed_vander)
            J_coeffs[i] = get_coeffs_1d_from_samples(dzs[i]/ks,mixed_vander)
        if i%10==0:
            fg = np.dot(np.dot(test_vander.T,I_coeffs[i]),zs[i])
            gg = np.dot(zs[i],zs[i])
            fit_check[i//10] = fg/gg
            fg = np.dot(np.dot(test_vander.T,J_coeffs[i]),dzs[i]/ks)
            gg = np.dot(dzs[i]/ks,dzs[i]/ks)
            dfit_check[i//10] = fg/gg
    t3 = time()
    print('# Time decomp loop:',t3-t2)
    sys.stdout.flush()

    I_coeffs_final = np.zeros( (l_max,len(t_zs)) ).T*1j
    J_coeffs_final = np.zeros( (l_max,len(t_zs)) ).T*1j
    print('# Alloc\'d final')
    num_exps_cs = config.num_exps_cs
    print('# Alloc\'d final1')
    in_basisA = set_up_flat_basis(config.k_min,config.k_max,num_exps_cs)
    print('# Alloc\'d final2')
    in_basisB = np.copy(basis_funcs)
    print('# Start\'n prod')
    prod = gen_series_product_matrix(in_basisA,in_basisB,basis_funcs,config.k_min,config.k_max)
    print('# Alloc\'d prod')
    exp_fit_check = np.zeros(len(t_zs)//10+1)*1j
    dexp_fit_check = np.zeros(len(t_zs)//10+1)*1j
    test_vander = eval_vander(basis_funcs,config.k_min,config.k_max,kbars)
    gen_exp_time = 0
    gen_exp_time_b = 0
    fit_check_time = 0
    t4 = time()
    decomp_xs,decomp_ws = leggauss(LG_INV_FUNC_RES)
    zs = basis_funcs[0](decomp_xs)
    norm_0 = np.dot(zs**2,decomp_ws)
    rotate_inds = np.where(config.late_t_switch>t_zs)[0]
    non_rotate_inds = np.where(config.late_t_switch<=t_zs)[0]
    t4a = time()
    print('# Starting gen_exp')
    exp_cs = gen_exp_coeffs(config.k_min,config.k_max,tau_s_array[rotate_inds],num_exps_cs,norm_0)
    t4b = time()
    gen_exp_time += t4b-t4a
    batch_size = 100
    batch_breaks = np.arange(0,len(rotate_inds),batch_size)[1:]
    t4c = time()
    batches = np.split(rotate_inds,batch_breaks)
    print('# Starting batches:', batch_size, flush=True)
    for batch in batches:
        I_coeffs_final[batch, :] = np.einsum('pi,pj,jik->pk', I_coeffs[batch], exp_cs[batch], prod, optimize=True)
        J_coeffs_final[batch, :] = np.einsum('pi,pj,jik->pk', J_coeffs[batch], exp_cs[batch], prod, optimize=True)
    print('# Done batches', flush=True)
    t4d = time()
    gen_exp_time_b += t4d-t4c
    I_coeffs_final[non_rotate_inds,:] = I_coeffs[non_rotate_inds]
    J_coeffs_final[non_rotate_inds,:] = J_coeffs[non_rotate_inds]
    t5 = time()
    print('# Time rotate loop:',t5-t4)
    print('# of which gen_exp:',gen_exp_time)
    print('# of which non exp:',gen_exp_time_b, flush=True)
    sys.stdout.flush()

    tf = time()
    print('# Time total decomp_zs:',tf-t0, flush=True)
    sys.stdout.flush()
    len_check = min(len(t_zs[::10]),len(fit_check.real))
    to_print = np.array([t_zs[::10][:len_check],fit_check.real[:len_check],fit_check.imag[:len_check],dfit_check.real[:len_check],dfit_check.imag[:len_check],exp_fit_check.real[:len_check],exp_fit_check.imag[:len_check],dexp_fit_check.real[:len_check],dexp_fit_check.imag[:len_check]]).T
    return I_coeffs_final.T,J_coeffs_final.T,to_print

def construct_and_integrate(t_zs, I_coeffs, J_coeffs, bkgd_interps, basis_funcs, l_max, final_basis):
    t0 = time()
    t1 = time()
    eps_array     = bkgd_interps[1](t_zs)
    eps_s_array   = bkgd_interps[2](t_zs)
    eta_array     = bkgd_interps[3](t_zs)
    phi_array     = bkgd_interps[6](t_zs)
    c_s_array     = bkgd_interps[7](t_zs)
    H_array       = bkgd_interps[8](t_zs)
    tau_s_array   = bkgd_interps[9](t_zs)
    cheat = np.ones(len(t_zs))*(1+0j)
    cheat[t_zs<config.N_start_integ] *= np.exp(-config.beta_activation*np.abs(tau_s_array[t_zs<config.N_start_integ]-tau_s_array[t_zs<config.N_start_integ][-1])**2)
    cheat[t_zs<config.N_start_integ-config.beta_margin] *= 0
    tau_s_simps_weights = simps_osc_weights_lg(tau_s_array,3*config.k_max)*np.exp(t_zs)/c_s_array

    cheat *= tau_s_simps_weights*np.exp(-3j*config.k_max*tau_s_array)
    t2 = time()
    print('# Time bkgd1:',t2-t1)
    sys.stdout.flush()
    bkgd_coeffs_0 = config.bkgd_coeff(t_zs,H_array,eps_array,eta_array,c_s_array,eps_s_array,0,phi_array)*cheat
    bkgd_coeffs_1 = config.bkgd_coeff(t_zs,H_array,eps_array,eta_array,c_s_array,eps_s_array,1,phi_array)*cheat
    bkgd_coeffs_2 = config.bkgd_coeff(t_zs,H_array,eps_array,eta_array,c_s_array,eps_s_array,2,phi_array)*cheat
    bkgd_coeffs_3 = config.bkgd_coeff(t_zs,H_array,eps_array,eta_array,c_s_array,eps_s_array,3,phi_array)*cheat
    bkgd_coeffs_4 = config.bkgd_coeff(t_zs,H_array,eps_array,eta_array,c_s_array,eps_s_array,4,phi_array)*cheat
    bkgd_coeffs_5 = config.bkgd_coeff(t_zs,H_array,eps_array,eta_array,c_s_array,eps_s_array,5,phi_array)*cheat
    bkgd_coeffs_list = [bkgd_coeffs_0,bkgd_coeffs_1,bkgd_coeffs_2,bkgd_coeffs_3,bkgd_coeffs_4,bkgd_coeffs_5]
    IJ_list = [[1,1,1],[1,1,0],[0,0,0],[1,1,0],[1,1,0],[0,0,1]]
    t3 = time()
    print('# Time bkgd2:',t3-t2)
    sys.stdout.flush()
    coeffs = np.zeros((l_max,l_max,l_max))

    assert len(bkgd_coeffs_0)==np.shape(I_coeffs)[1]
    assert len(bkgd_coeffs_1)==np.shape(J_coeffs)[1]
    maptimer = np.zeros(2)
    intgd_smpls = 10
    to_print_labels = ['t','tau']
    early_print_inds = np.where((t_zs>config.N_start_integ-0.1)*(t_zs<config.N_start_integ+0.1))[0]#[::intgd_smpls]
    to_print_slice = slice(max(-len(t_zs),-3000*intgd_smpls),None,intgd_smpls)
    late_print_inds = np.arange(len(t_zs))[to_print_slice]
    full_print_inds = np.array(list(early_print_inds)+list(late_print_inds),dtype=int)
    to_print = [t_zs[full_print_inds],tau_s_array[full_print_inds]]

    for s in range(len(bkgd_coeffs_list)):
        for i in range(len(basis_funcs)):
            temp = np.copy(bkgd_coeffs_list[s])[full_print_inds]
            if IJ_list[s][0]==0:
                temp *= I_coeffs[i,:][full_print_inds]
            else:
                temp *= J_coeffs[i,:][full_print_inds]
            if IJ_list[s][1]==0:
                temp *= I_coeffs[i,:][full_print_inds]
            else:
                temp *= J_coeffs[i,:][full_print_inds]
            if IJ_list[s][2]==0:
                temp *= I_coeffs[i,:][full_print_inds]
            else:
                temp *= J_coeffs[i,:][full_print_inds]
            temp *= np.exp(3j*config.k_max*tau_s_array[full_print_inds]*(t_zs[full_print_inds]>config.N_start_integ+1))
            temp /= tau_s_simps_weights[full_print_inds]
            to_print.append( np.copy(temp.imag) )
            #to_print_labels.append(f'{s}-{i}-{i}-{i}')
            to_print_labels.append(str(s)+'-'+str(i)+'-'+str(i)+'-'+str(i))
    t4 = time()
    coeff_results = np.zeros((len(config.shape_indices),l_max,l_max,l_max))
    temp1I = np.zeros(len(bkgd_coeffs_0))*1.j
    temp1J = np.zeros(len(bkgd_coeffs_0))*1.j
    temp2 = np.zeros(len(bkgd_coeffs_0))*1.j
    final_part = np.zeros_like(I_coeffs)*1.j
    for raw_ind, bkgd_vals in enumerate(bkgd_coeffs_list):
        print('# Sum of bkgd', raw_ind, '=', np.sum(bkgd_vals), flush=True)
    quad_sum.do_the_integrals(I_coeffs, J_coeffs, coeff_results, config.shape_indices, bkgd_coeffs_0, bkgd_coeffs_1, bkgd_coeffs_2, bkgd_coeffs_3, bkgd_coeffs_4, bkgd_coeffs_5, temp1I, temp1J, temp2, final_part)

    coeffs = coeff_results.reshape(len(config.shape_indices),l_max,l_max,l_max)
    raw_coeffs = np.copy(coeffs)
    for raw_ind, raw_cs in enumerate(raw_coeffs):
        print('# Sum of raw', config.shape_indices[raw_ind], '=', np.sum(raw_cs), flush=True)
    t5 = time()
    print('# Time map     :',t5-t4)
    print('# of which eval:',maptimer[0])
    print('# of which simp:',maptimer[1])
    sys.stdout.flush()
    for i_s in range(len(config.shape_indices)):
        for j in range(l_max):
            for i in range(j):
                coeffs[i_s,i,j] = coeffs[i_s,j,i]

    t6 = time()
    tidied_coeffs, basis_funcs_padded = tidy_coeffs(coeffs, basis_funcs, config.k_min, config.k_max, final_basis)
    t7 = time()
    print('# Time tidying:',t7-t6)
    sys.stdout.flush()
    tf = time()
    print('# Time total integr:',tf-t0)
    print('# Time: ',t7-t6,t6-t5,t5-t4,t4-t3,t3-t2,t2-t1,t1-t0)
    sys.stdout.flush()
    return tidied_coeffs, basis_funcs_padded, np.array(to_print).T, ', '.join(to_print_labels), raw_coeffs

def augment_basis_arb_func(k_min,k_max,basis,func,Nk=Nk_aug):
    klim = [k_min,k_max]
    xs,ws = leggauss(Nk)
    norms = np.zeros(len(basis))
    for i,b in enumerate(basis):
        fs = b(xs)
        norms[i] = np.sqrt(np.dot(fs**2,ws))
    uxs = 0.5*(xs*(klim[1]-klim[0])+(klim[1]+klim[0]))
    f_vals = func(uxs)
    vander = np.array([b(xs) for b in basis])
    #cs = np.sum(ws*vander*f_vals,axis=1)/norms
    cs = np.zeros_like(basis, dtype=np.float64)
    cs[0] = np.dot(ws*vander[0],f_vals)
    f_hat = np.copy(f_vals)
    for i in range(1,len(basis)):
        f_hat = f_hat - cs[i-1]*vander[i-1]
        cs[i] = np.dot(ws*vander[i],f_hat)
    new_basis = [None]*(len(basis)+1)
    new_basis[1:] = basis
    f_orth_vals = f_vals-np.sum(cs*vander.T,axis=1)
    norm = np.sqrt(np.sum(ws*f_orth_vals**2))
    if norm<1e-6:
        print('Norm is suspiciously small:',norm)
        #return None
    def f_orth(x):
        ux = 0.5*(x*(klim[1]-klim[0])+(klim[1]+klim[0]))
        vander = np.array([b(x) for b in basis])
        return (func(ux)-np.sum((cs*vander.T).T,axis=0))/norm
    new_basis[0] = f_orth
    return new_basis

def set_up_mixed_basis(k_min, k_max, Nb, n_s_mod=-0.0325, verbose=True, quad=False, extra=None, flat=False):
    if flat:
        basis_funcs = set_up_flat_basis(k_min, k_max, Nb)
        if verbose:
            print('# Flat')
            print('# Ortho check:',Nb,'=', np.sum(check_orthog(basis_funcs,k_min,k_max)), flush=True)
        return basis_funcs
    def inv_func(x):
        return x**-(1-n_s_mod)
    def inv_func2(x):
        return x**-(2-n_s_mod)#((1-n_s_mod)/3.)
    aug_funcs = [inv_func]
    if quad:
        aug_funcs.append(inv_func2)
    if extra is not None:
        aug_funcs.append(extra)
    base_basis = set_up_flat_basis(k_min,k_max,Nb-len(aug_funcs))
    basis_funcs = np.copy(base_basis)
    for i,af in enumerate(aug_funcs):
        basis_funcs = augment_basis_arb_func(k_min,k_max,basis_funcs,af,Nk=Nk_aug)
        if verbose:
            print('# Aug func ln(f(2))/ln(2)', i, ':', np.log(af(2.))/np.log(2.))
            #print('# Aug func',i,':', af.__name__)
    if verbose:
        print('# n_s_mod:',n_s_mod)
        print('# Ortho check:',Nb,'=', np.sum(check_orthog(basis_funcs,k_min,k_max)), flush=True)
    return basis_funcs

def set_up_log_basis(k_min, k_max, Nb, inv=False, verbose=True):
    if not inv:
        aug_funcs = [lambda x: np.log(x)]
    else:
        aug_funcs = [lambda x: 1./x, lambda x: np.log(x)/x]
    base_basis = set_up_flat_basis(k_min,k_max,Nb-len(aug_funcs))
    basis_funcs = np.copy(base_basis)
    for i,af in enumerate(aug_funcs):
        basis_funcs = augment_basis_arb_func(k_min,k_max,basis_funcs,af,Nk=Nk_aug)
    if verbose:
        print('# inv:', inv)
        print('# Ortho check:',Nb,'=', np.sum(check_orthog(basis_funcs,k_min,k_max)), flush=True)
    return basis_funcs

