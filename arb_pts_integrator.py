import numpy as np
from numpy.polynomial.legendre import legvander

def second_order_lg_weights(xs):
    '''
    Integration weights for given sample points,
    using a second-order Legendre polynomial
    expansion for each segment [x_{i-1}, x_{i+1}]
    for i=1, 3, ..., N-1, if N is odd.
    If the number of points is even,
    the final segment [x_{N-1}, x_{N}]
    is dealt with using a first-order expansion.
    Once the weights have been calculated for
    a given set of points, the integral of
    any sufficiently smooth function (f) over those
    points (xs) is simply:
    integral = np.dot(weights, f(xs))
    giving a cheap way of integrating many functions
    over the same fixed set of sample points.

    Examples
    --------
    >>> import numpy as np
    >>> dec_places = 5
    >>> ## # Evenly spaced points
    >>> N = 1000
    >>> xs = np.linspace(0, 1, N, endpoint=True)
    >>> weights = second_order_lg_weights(xs)
    >>> f = lambda x: 2*np.pi**2*x**2*np.cos(2*np.pi*x)
    >>> res = np.dot(weights, f(xs))
    >>> round(res, dec_places)
    1.0
    >>> ## # Randomly spaced points, and the endpoints.
    >>> N = 1000
    >>> np.random.seed(28)
    >>> xs = np.array([0]+list(np.sort(np.random.random(N-2)))+[1])
    >>> weights = second_order_lg_weights(xs)
    >>> f = lambda x: 2*np.pi**2*x**2*np.cos(2*np.pi*x)
    >>> res = np.dot(weights, f(xs))
    >>> round(res, dec_places)
    1.0
    >>> ## # Odd number of randomly spaced points, and the endpoints.
    >>> N = 999
    >>> np.random.seed(28)
    >>> xs = np.array([0]+list(np.sort(np.random.random(N-2)))+[1])
    >>> weights = second_order_lg_weights(xs)
    >>> f = lambda x: 2*np.pi**2*x**2*np.cos(2*np.pi*x)
    >>> res = np.dot(weights, f(xs))
    >>> round(res, dec_places)
    1.0
    '''
    xs = np.copy(np.array(xs))
    weights = np.zeros(len(xs))
    for i in range(1, len(xs)-1, 2):
        ## # Take 3 points.
        xx = xs[i-1:i+2]
        ## # Map to -1, something, 1
        xxb = (2*xx-(xx[0]+xx[2]))/(xx[2]-xx[0])
        lV = legvander(xxb, 2)
        ## # Invert to get the coefficients.
        ilV = np.linalg.inv(lV)*(xx[2]-xx[0])
        ## # Then the 0 coeff is prop to the definite integral.
        weights[i-1:i+2] += ilV[0]
    if len(xs)%2 == 0:
        ## # Include leftover 2 points.
        xx = xs[-2:]
        xxb = np.array([-1, 1])
        lV = legvander(xxb, 1)
        ilV = np.linalg.inv(lV)
        fix = ilV[0]*(xx[1]-xx[0])
        weights[-2] += fix[0]
        weights[-1] += fix[1]
    return weights

def gen_osc_weights(T, order):
    '''
    Integral of legendre(order, x)*exp(i*x*T) from -1 to 1

    Examples
    --------
    >>> import numpy as np
    >>> round(gen_osc_weights(np.pi, 2), 6)
    -0.607927
    >>> round(gen_osc_weights(np.pi/2., 2), 6)
    -0.274834
    >>> res = gen_osc_weights(0.5, 1)
    >>> round(res.imag, 6)
    0.325074
    >>> round(res.real, 6)
    0.0
    >>> round(gen_osc_weights(np.pi/2., 0), 5)
    1.27324
    '''
    assert order in [0, 1, 2]
    if order == 0:
        return 2*np.sinc(T/np.pi)
    elif order == 1:
        if np.abs(T) > 1e-6:
            return -2j*(T*np.cos(T)-np.sin(T))*T**-2
        else:
            return -2j*(-T/3.+T**3/30.-T**5/840.)
    elif order == 2:
        if np.abs(T) > 1e-6:
            return (6*T*np.cos(T)+(2*T**2-6)*np.sin(T))*T**-3
        else:
            return -2.*(T**2)/15+T**4/105.-T**6/3780.

def simps_osc_weights_lg(xs, t):
    '''
    Integration weights for given sample points,
    with oscillatory weight function exp(t*1i*x),
    using a second-order Legendre polynomial
    expansion for each segment [x_{i-1}, x_{i+1}]
    for i=1, 3, ..., N-1, if N is odd.
    If the number of points is even,
    the final segment [x_{N-1}, x_{N}]
    is dealt with using a first-order expansion.
    Once the weights have been calculated for
    a given set of points, the integral of
    any sufficiently smooth function (f) multiplied
    by exp(t*1i*x) over those points (xs) is simply:
    integral = np.dot(weights, f(xs))
    giving a cheap way of integrating many functions
    over the same fixed set of sample points and
    the same oscillatory weight function.

    Examples
    --------
    >>> ## # Randomly spaced points, and the endpoints.
    >>> N = 1000
    >>> dec_places = 5
    >>> np.random.seed(28)
    >>> xs = np.array([0]+list(np.sort(np.random.random(N-2)))+[1])
    >>> weights = simps_osc_weights_lg(xs, 2*np.pi)
    >>> f = lambda x: 2*np.pi**2*x**2
    >>> res = np.dot(weights, f(xs))
    >>> round(res.real, dec_places)
    1.0
    >>> round(res.imag, dec_places)
    -3.14159
    >>> f = lambda x: x**3
    >>> weights = simps_osc_weights_lg(xs, 10)
    >>> res = np.dot(weights, f(xs))
    >>> round(res.real, dec_places)
    -0.07521
    >>> round(res.imag, dec_places)
    0.06288
    >>> ## # Same examples, with an odd number of points.
    >>> N = 1001
    >>> dec_places = 7
    >>> np.random.seed(28)
    >>> xs = np.array([0]+list(np.sort(np.random.random(N-2)))+[1])
    >>> weights = simps_osc_weights_lg(xs, 2*np.pi)
    >>> f = lambda x: 2*np.pi**2*x**2
    >>> res = np.dot(weights, f(xs))
    >>> round(res.real, dec_places)
    1.0
    >>> round(res.imag, dec_places)
    -3.1415927
    >>> f = lambda x: x**3
    >>> weights = simps_osc_weights_lg(xs, 10)
    >>> res = np.dot(weights, f(xs))
    >>> round(res.real, dec_places)
    -0.0752067
    >>> round(res.imag, dec_places)
    0.0628785
    '''
    xs = np.copy(xs)
    weights = np.zeros(len(xs))*1j
    for i in range(1, len(xs)-1, 2):
        ## # Take 3 points.
        xx = xs[i-1:i+2]
        ## # Map to -1, something, 1
        xxb = (2*xx-(xx[0]+xx[2]))/(xx[2]-xx[0])
        lV = legvander(xxb, 2)
        ## # Invert to get the coefficients.
        ilV = np.linalg.inv(lV)
        ## # Integrals of Legendre polynomials*exp(1j*t*x)
        osc_weights = np.zeros(3)*1j
        T = t*(xx[2]-xx[0])/2.
        osc_weights[0] = gen_osc_weights(T, 0)
        osc_weights[1] = gen_osc_weights(T, 1)
        osc_weights[2] = gen_osc_weights(T, 2)
        ## # Undoing the mapping to [-1, 1]
        osc_I = 0.5*(xx[2]-xx[0])*(np.exp(1j*t*(xx[2]+xx[0])/2.)*osc_weights)
        weights[i-1:i+2] += np.dot(osc_I, ilV)
    if len(xs)%2 == 0:
        ## # Include leftover 2 points.
        xx = xs[-2:]
        xxb = np.array([-1, 1])
        osc_weights = np.zeros(2)*1j
        T = t*(xx[1]-xx[0])/2.
        osc_weights[0] = gen_osc_weights(T, 0)
        osc_weights[1] = gen_osc_weights(T, 1)
        osc_I = 0.5*(xx[1]-xx[0])*(np.exp(1j*t*(xx[1]+xx[0])/2.)*osc_weights)
        lV = legvander(xxb, 1)
        ilV = np.linalg.inv(lV)
        fix = np.dot(osc_I, ilV)
        weights[-2] += fix[0]
        weights[-1] += fix[1]
    return weights

if __name__ == "__main__":
    import doctest
    doctest.testmod()
