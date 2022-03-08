'''
Efficient evaluation of the equations of motion, using splines and cython.
'''

cimport cython
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef int searchsorted(double[:] xs,double x) nogil:
    cpdef int lo = 0
    cpdef int hi = xs.shape[0]-1
    if x<xs[1]:
        return 0
    else:
        while hi>lo+1:
            mi = (lo+hi)//2
            if xs[mi]>=x:
                hi = mi
            else:
                lo = mi
        return hi

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef int get_spl_index(double x,double[:] xs) nogil:
    cpdef int ind = searchsorted(xs,x)-1
    ind = ind*(ind!=-1)
    return ind

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef double spline_eval(int ind,double x,double[:] A,double[:] B) nogil:
    return A[ind]+x*B[ind]

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef double[:] mode_evolution_cython(double[:] y,double t,double[:] ks,int threshold_early,int threshold_late,double[:,:] As,double[:,:] Bs,double[:] Ns, double[:] dy ) nogil:
    cpdef int Nk = ks.shape[0]
    #cpdef np.ndarray[double, ndim=1] dy = np.zeros(4*Nk)
    cpdef double temp_1,temp_2
    #y =[[R_re], [R_im], [R'_re], [R'_im]]
    cpdef int ind = get_spl_index(t,Ns)
    cpdef double dtau_s = spline_eval(ind,t,As[0],Bs[0])
    cpdef double eps = spline_eval(ind,t,As[1],Bs[1])
    cpdef double eps_s = spline_eval(ind,t,As[2],Bs[2])
    cpdef double eta = spline_eval(ind,t,As[3],Bs[3])
    cpdef double P = spline_eval(ind,t,As[4],Bs[4])
    cpdef double Q = spline_eval(ind,t,As[5],Bs[5])

    cpdef int i
    for i in range(0,Nk):
        dy[i] = y[i+2*Nk]
        dy[i+Nk] = y[i+3*Nk]

    #y =[[R_re], [R_im], [R'_re], [R'_im]]
    for i in range(threshold_late,threshold_early):
        dy[i+2*Nk] = -(3-eps)*y[i+2*Nk] - 2*ks[i]*dtau_s*y[i+3*Nk]
        dy[i+2*Nk] += -y[i]*P
        dy[i+2*Nk] += +y[i+Nk]*Q*ks[i]*dtau_s

        dy[i+3*Nk] = -(3-eps)*y[i+3*Nk] + 2*ks[i]*dtau_s*y[i+2*Nk]
        dy[i+3*Nk] += -y[i+Nk]*P
        dy[i+3*Nk] += -y[i]*Q*ks[i]*dtau_s

    temp_1 = (3-eps+eta-2*eps_s)
    for i in range(0,threshold_late):
        temp_2 = (ks[i]*dtau_s)**2
        dy[i+2*Nk] = -temp_1*y[i+2*Nk]-temp_2*y[i]
        dy[i+3*Nk] = -temp_1*y[i+3*Nk]-temp_2*y[i+Nk]
    return dy

