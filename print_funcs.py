'''
Functions to print and plot useful quantities, for example
checking against the consistency condition.
'''

import numpy as np
from orthog_mixed_basis_3d import tidied_eval,corr,print_tetra,get_coeffs_1d_from_samples,eval_vander,basis_vander,gen_series_product_matrix,gen_exp_coeffs,tidy_coeffs

def bar(x,k_min,k_max):
    return (2*x-(k_min+k_max))/(k_max-k_min)

def unbar(x,k_min,k_max):
    return 0.5*((k_max-k_min)*x+k_min+k_max)

def in_tetra(p,lower_lim=0):
    x,y,z = p
    return ( (x+y>=z) and (y+z>=x) and (z+x>=y) and (x+y+z>lower_lim) )

def tidied_coeff_print(k_min, k_max, coeffs, basis_funcs, f_name, basis_params=[]):
    data_file = open(f_name,'w')
    l = len(coeffs[0,0,:])
    data_file.write( '"i","j","k","a_ijk"#'+str(k_min)+','+str(k_max)+','+','.join([str(x) for x in basis_params])+'\n' )
    for i in range(l):
        for j in range(l):
            for k in range(l):
                data_file.write( str(i)+','+str(j)+','+str(k)+','+str(coeffs[i,j,k])+'\n' )
            data_file.write( '\n' )
        data_file.write( '\n' )
    data_file.close()

# no PS check
def tidied_sqz_print(k_min,k_max,k_mark,coeffs,basis_funcs,f_name,compare_func=0):
    data_file = open(f_name,'w')
    ks = np.linspace(k_min,k_mark,10000)
    # Starting from flattened, across eql, into sqz. Staying on K=2*k_mark
    points = np.array([[k_mark-0.5*k,k_mark-0.5*k,k] for k in ks]).T
    points = points[:,np.apply_along_axis(in_tetra,0,points)]
    xs = points[0]
    ys = points[1]
    zs = points[2]
    decomp = tidied_eval(k_min,k_max,coeffs,basis_funcs,points)*zs/ys
    data_file.write( 'k1,k2,k3,S\n' )
    for i,r in enumerate(decomp):
        data_file.write( str(xs[i])+','+str(ys[i])+','+str(zs[i])+','+str(r))
        if compare_func!=0:
                data_file.write( ','+str(compare_func(xs[i],ys[i],zs[i])) )
        data_file.write( '\n' )	
    data_file.close()

def tidied_sqz_check(k_min,k_max,k_mark,coeffs,basis_funcs,ps_eval,f_name):
    data_file = open(f_name,'w')
    dps_eval = ps_eval.derivative()
    ks = np.linspace(k_min,k_mark,10000)
    # Starting from flattened, across eql, into sqz. Staying on K=2*k_mark
    points = np.array([[k_mark-0.5*k,k_mark-0.5*k,k] for k in ks]).T
    points = points[:,np.apply_along_axis(in_tetra,0,points)]
    xs = points[0]
    ys = points[1]
    zs = points[2]
    decomp = tidied_eval(k_min,k_max,coeffs,basis_funcs,points)
    kS = np.sqrt(0.5*(xs**2+ys**2)-0.25*zs**2)
    ps_side = -dps_eval(kS)*np.exp(ps_eval(kS)+ps_eval(zs))*xs**2*ys**2/(zs*kS**2)
    data_file.write( 'k1,k2,k3,S,ps-side,k3/k1,(n_s-1)k,S/PP\n' )
    for i,r in enumerate(decomp):
        data_file.write( str(xs[i])+','+str(ys[i])+','+str(zs[i])+','+str(r)+','+str(ps_side[i])+','+str(zs[i]/xs[i])+','+str(dps_eval(xs[i]))+','+str(-r/(np.exp(ps_eval(xs[i])+ps_eval(zs[i]))*ys[i]))+'\n' )
    data_file.close()

def along_edge_sqz_check(k_min,k_max,sqz_ratio,coeffs,basis_funcs,ps_eval,f_name):
    k_mark = 0.5*(k_min+k_max)
    data_file = open(f_name,'w')
    dps_eval = ps_eval.derivative()
    ks = np.linspace(k_min,k_mark,10000)
    points = np.array([[k,k,k*sqz_ratio] for k in ks]).T
    points = points[:,np.apply_along_axis(in_tetra,0,points)]
    xs = points[0]
    ys = points[1]
    zs = points[2]
    decomp = tidied_eval(k_min,k_max,coeffs,basis_funcs,points)*zs/ys
    ps_side = -dps_eval(xs)*np.exp(ps_eval(xs)+ps_eval(zs))*ys
    data_file.write( 'k1,k2,k3,S,ps-side,k3/k1,(n_s-1)k,S/PP\n' )
    for i,r in enumerate(decomp):
        data_file.write( str(xs[i])+','+str(ys[i])+','+str(zs[i])+','+str(r)+','+str(ps_side[i])+','+str(zs[i]/xs[i])+','+str(dps_eval(xs[i]))+','+str(-r/(np.exp(ps_eval(xs[i])+ps_eval(zs[i]))*ys[i]))+'\n' )
    data_file.close()


def tidied_eql_print(k_min,k_max,coeffs,basis_funcs,f_name,compare_func=0):
    data_file = open(f_name,'w')
    ks = np.linspace(k_min,k_max,10000)
    points = np.array([[k,k,k] for k in ks]).T
    decomp = tidied_eval(k_min,k_max,coeffs,basis_funcs,points)
    data_file.write( 'k1,k2,k3,S\n' )
    for i,r in enumerate(decomp):
        data_file.write( str(ks[i])+','+str(ks[i])+','+str(ks[i])+','+str(r) )
        if compare_func!=0:
                data_file.write( ','+str(compare_func(ks[i],ks[i],ks[i])) )
        data_file.write( '\n' )	
    data_file.close()

def tidied_various_lines_print(k_min,k_max,coeffs,basis_funcs,f_name,ps_eval=None,mask=lambda x,y,z:1.):
    if ps_eval is None:
        def ps_eval(k): return 0
        def dps_eval(k): return 0
    else:
        dps_eval = ps_eval.derivative()
    data_file = open(f_name,'w')
    ks = np.linspace(k_min,k_max,10000)
    sqznesses = [2., 1., 0.01, 2*k_min/k_max]
    for sqz in sqznesses:
        sqz_points = np.array([[k,k,sqz*k] for k in ks[(ks>k_min/sqz)*(ks<k_max/sqz)]]).T
        sqz_decomp = tidied_eval(k_min,k_max,coeffs,basis_funcs,sqz_points)*mask(*sqz_points)
        xs,ys,zs = sqz_points
        kS = np.sqrt(0.5*(xs**2+ys**2)-0.25*zs**2)
        ps_side = -dps_eval(kS)*np.exp(ps_eval(kS)+ps_eval(zs))*xs**2*ys**2/(zs*kS**2)
        data_file.write( "x,y,z,S:"+str(sqz)+',PS\n' )
        for i in range(len(sqz_decomp)):
            data_file.write( str(xs[i])+',' )	
            data_file.write( str(ys[i])+',' )	
            data_file.write( str(zs[i])+',' )	
            data_file.write( str(sqz_decomp[i])+',' )
            data_file.write( str(ps_side[i]) )
            data_file.write( '\n' )
        data_file.write( '\n\n' )

def triangle_print_to_file(k_min,k_max,p,coeffs,basis_funcs,p1,p2,p3,data_file,Np=100):
    p1,p2,p3 = np.array([p1,p2,p3])
    for s in np.linspace(0,1,Np,endpoint=True):
        a = p3*s+p1*(1-s)
        b = p3*s+p2*(1-s)
        points = [a*t+b*(1-t) for t in np.linspace(0,1,1+Np*(1-s))]
        points = np.array(points).T
        decomp = tidied_eval(k_min,k_max,coeffs,basis_funcs,points)
        for i,r in enumerate(decomp):
            data_file.write( str(points[0][i])+','+str(points[1][i])+','+str(points[2][i])+','+str(r)+'\n' )
        data_file.write( '\n' )

def print_faces(k_min,k_max,coeffs,basis_funcs,f_name):
    shift = 0#(k_max-k_min)*0.2
    sk_min = k_min+shift*2./3.
    xtop = [k_min+shift,k_max-shift,k_max-shift]
    ytop = [k_max-shift,k_min+shift,k_max-shift]
    ztop = [k_max-shift,k_max-shift,k_min+shift]
    origin = [sk_min,sk_min,sk_min]
    middle_face = [0.5*(k_min+k_max),0.5*(k_min+k_max),k_max]
    data_file = open(f_name,'w')
    data_file.write( '"X","Y","Z","f(X,Y,Z)"\n' )
    triangle_print_to_file(k_min,k_max,coeffs,basis_funcs,middle_face,ytop,ztop,data_file,Np=500)
    data_file.write( '\n' )
    triangle_print_to_file(k_min,k_max,coeffs,basis_funcs,middle_face,ytop,origin,data_file,Np=500)
    data_file.write( '\n' )
    triangle_print_to_file(k_min,k_max,coeffs,basis_funcs,ztop,middle_face,origin,data_file,Np=500)
    data_file.write( '\n' )
    triangle_print_to_file(k_min,k_max,coeffs,basis_funcs,ztop,ytop,origin,data_file,Np=500)
    data_file.close()

def ps_planck_compar(k_min, k_max, ps_eval, dps_eval, f_name, k_pivot=None):
    if k_pivot is None:
        ## # Should this be the geometric mean?
        k_pivot = 0.5*(k_min+k_max)
    ## # 3.044 ± 0.014
    ln10A_s = 3.044
    ## # 0.965 ± 0.004
    n_s = 0.965
    data_file = open(f_name, 'w')
    data_file.write('"ln(k)","ln(1e10*k^3*P(k)/(2pi^2))","Planck PS","n_s-1","Planck n_s-1"#'+str(k_min)+','+str(k_max)+'\n')
    for logk in np.linspace(np.log(k_min), np.log(k_max), 400):
        line = str(logk)+','
        line += str(ps_eval(np.exp(logk))+np.log(1e10/(2*np.pi**2)))+','
        line += str(ln10A_s+(n_s-1.)*(logk-np.log(k_pivot)))+','
        line += str(np.exp(logk)*dps_eval(np.exp(logk)))+','
        line += str(n_s-1.)
        line += '\n'
        data_file.write(line)
    data_file.close()

def print_ps_in_w_ks(f_name, k_min, k_max, k_pivot, ps_eval, dps_eval, H, eps, c_s, eta, eps_s, N_k):
    ## # 3.044 ± 0.014
    ln10A_s = 3.044
    ## # 0.9649 ± 0.0042
    n_s = 0.9649
    PR = "\mathcal{P}_{\mathcal{R}}"
    labels = [r"$ln(k)$", r"$ln(10^{10}"+PR+")$", r"Planck $"+PR+"$", r"SR $"+PR+"$"]
    labels += [r"$n_s$", r"Planck $n_s$", r"SR $n_s$"]
    logks = np.linspace(np.log(k_min), np.log(k_max), 400)
    planck_10ps = ln10A_s+(n_s-1)*(logks-np.log(k_pivot))
    Ns = N_k(np.exp(logks))
    sr_ps = H(Ns)**2/(8*np.pi**2*eps(Ns)*c_s(Ns))
    sr_ns = 1-2*eps(Ns) - eta(Ns) - eps_s(Ns)
    data_file = open(f_name, 'w')
    data_file.write(','.join(labels)+'\n')
    for i, logk in enumerate(logks):
        line = str(logk)+','
        line += str(ps_eval(np.exp(logks[i]))+np.log(1e10/(2*np.pi**2)))+','
        line += str(planck_10ps[i])+','
        line += str(np.log(1e10*sr_ps[i]))+','
        line += str(1+np.exp(logks[i])*dps_eval(np.exp(logks[i])))+','
        line += str(n_s)+','
        line += str(sr_ns[i])
        line += '\n'
        data_file.write(line)
    data_file.close()

