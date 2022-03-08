'''
Assemble the integrands (with the integration weights already included in the background factors)
and perform the dot products.
'''

cimport cython
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef double quad_sum(double[:] arr1, double complex[:] arr2, double complex[:] arr3, double complex[:] arr4) nogil:
    cdef size_t i, I
    #cdef double total = 0
    cdef double complex total = 0.+0.j
    I = arr1.shape[0]
    for i in range(I):
        #total += (arr1[i]*arr2[i]*arr3[i]*arr4[i]).imag
        total += arr1[i]*arr2[i]*arr3[i]*arr4[i]
    return total.imag

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef double double_sum(double complex[:] arr1, double complex[:] arr2) nogil:
    cdef size_t i, I
    cdef double complex total = 0.+0.j
    I = arr1.shape[0]
    for i in range(I):
        total += arr1[i]*arr2[i]
    return total.imag

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef void double_complex_prod(double complex[:] arr1, double complex[:] arr2, double complex[:] result) nogil:
    cdef size_t i, I
    I = arr1.shape[0]
    for i in range(I):
        result[i] = arr1[i]*arr2[i]

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef void double_prod(double complex[:] arr1, double complex[:] arr2, double complex[:] result) nogil:
    cdef size_t i, I
    I = arr1.shape[0]
    for i in range(I):
        result[i] = arr1[i]*arr2[i]
        
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef void do_the_integrals(double complex[:,:] I_coeffs, double complex[:,:] J_coeffs, double[:,:,:,:] coeff_results, int[:] shape_indices, double complex[:] bkgd_coeffs_0, double complex[:] bkgd_coeffs_1, double complex[:] bkgd_coeffs_2, double complex[:] bkgd_coeffs_3, double complex[:] bkgd_coeffs_4, double complex[:] bkgd_coeffs_5, double complex[:] temp1I, double complex[:] temp1J, double complex[:] temp2, double complex[:,:] final_part) nogil:
    ## # At present this is no faster than the python implementation.
    ## # Is this because of the interior if statements?
    cdef size_t i, j, k, l_max, s_max
    s_max = shape_indices.shape[0]
    l_max = I_coeffs.shape[0]
    for i in range(l_max):
        for j in range(i+1):
            double_complex_prod(I_coeffs[i],I_coeffs[j],temp1I)
            double_complex_prod(J_coeffs[i],J_coeffs[j],temp1J)
            for s in range(s_max):
                if shape_indices[s]==0:
                    double_prod(bkgd_coeffs_0,temp1J,temp2)
                    final_part = J_coeffs
                if shape_indices[s]==1:
                    double_prod(bkgd_coeffs_1,temp1J,temp2)
                    final_part = I_coeffs
                if shape_indices[s]==2:
                    double_prod(bkgd_coeffs_2,temp1I,temp2)
                    final_part = I_coeffs
                if shape_indices[s]==3:
                    double_prod(bkgd_coeffs_3,temp1J,temp2)
                    final_part = I_coeffs
                if shape_indices[s]==4:
                    double_prod(bkgd_coeffs_4,temp1J,temp2)
                    final_part = I_coeffs
                if shape_indices[s]==5:
                    double_prod(bkgd_coeffs_5,temp1I,temp2)
                    final_part = J_coeffs
                for k in range(l_max):
                    #coeff_results[shape_indices[s],i,j,k] = double_sum(temp2,final_part[k])
                    coeff_results[s,i,j,k] = double_sum(temp2,final_part[k])

