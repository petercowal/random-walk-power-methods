import numpy as np
from collections import deque

def measure_error(x1, xtrue):
    if xtrue is None:
        return 1
    x1 = x1.flatten()
    xtrue = xtrue.flatten()
    return np.linalg.norm(np.vdot(x1,xtrue)/np.vdot(x1,x1)*x1-xtrue)/np.linalg.norm(xtrue)

# un-normalized implementation (simplest, but prone to overflow)
def momentum_general_unnormalized(A, v0, betas, n, xtrue = None):
    m = len(betas)
    errs = np.zeros(n + 1)
    xs = deque([v0])
    errs[0] = measure_error(v0, xtrue)
    for i in range(1, len(betas)):
        xs.append(A @ xs[-1])
        errs[i] = measure_error(xs[-1], xtrue)

    for i in range(len(betas), n + 1):
        xnew = A @ xs[-1]
        for j in range(len(betas)):
            xnew = xnew - betas[j]*xs[-j - 1]

        xs.append(xnew)
        xs.popleft()
        errs[i] = measure_error(xnew, xtrue)

    return xs[-1], errs

# normalized implementation
def momentum_general(A, v0, betas, n, p0 = 1, xtrue = None):
    m = len(betas)
    errs = np.zeros(n + 1)
    hs = deque([np.linalg.norm(v0)])
    xs = deque([v0/hs[0]])

    errs[0] = measure_error(v0, xtrue)
    for i in range(1, m):
        vnew = p0 * A @ xs[-1]
        hnew = np.linalg.norm(vnew)
        xnew = vnew/hnew
        hs.append(hnew)
        xs.append(xnew)
        errs[i] = measure_error(xs[-1], xtrue)

    for i in range(m, n + 1):
        vnew = A @ xs[-1]
        hprod = 1/hs[-1]

        for j in range(m):
            hprod *= hs[-j - 1]
            vnew -= betas[j]*xs[-j - 1]/hprod

        hnew = np.linalg.norm(vnew)
        xnew = vnew/hnew

        hs.append(hnew)
        hs.popleft()

        xs.append(xnew)
        xs.popleft()
        errs[i] = measure_error(xnew, xtrue)

    return xs[-1], errs

# direct inversion of convergence rate
def r_option_1(Cinv, rho):
    return 1/(Cinv*(1/rho - 1)**2 + 1)

# asymptotically equivalent inversion of convergence rate
# that seems to work better in practice
def r_option_2(Cinv, rho):
    return 1/(Cinv*np.log(rho)**2 + 1)

# dynamic algorithm
# implementation assumes that ps is a probability distribution
# satisfying the criteria described in our theorem
# for example, the dynamic chebyshev method would use
# ps = [1/2, 0, 1/2]
# corresponding to p_0 = 1/2 and p_2 = 1/2
def momentum_dynamic(A, v0, ps, n, xtrue = None, r_method = r_option_2):
    m = len(ps) - 1

    errs = np.zeros(n + 1)
    hs = deque([np.linalg.norm(v0)])
    xs = deque([v0/hs[0]])

    errs[0] = measure_error(v0, xtrue)

    # compute constant used in estimating the convergence rate
    Cinv = 0
    for j in range(2, len(ps)):
        Cinv += ps[j]*j*(j-1)/2

    # initial power iterations
    for i in range(1, m):
        vnew = ps[0] * A @ xs[-1]
        hnew = np.linalg.norm(vnew)
        xnew = vnew/hnew
        hs.append(hnew)
        xs.append(xnew)
        errs[i] = measure_error(xs[-1], xtrue)

    nu = np.vdot(vnew,xs[-2])
    d1 = np.linalg.norm(vnew - nu*xs[-2])

    for i in range(m, n + 1):
        vnew = A @ xs[-1]

        nu = np.vdot(vnew,xs[-1])
        d2 = np.linalg.norm(vnew - nu*xs[-1])
        rho = np.min((d2/d1, 1))
        r = r_method(Cinv, rho)
        lambda2 = r*nu

        hprod = 1/hs[-1]
        # apply momentum terms
        for j in range(m):
            # estimate optimal parameter using dynamic method
            beta = ps[j + 1] * ps[0]**j * lambda2**(j + 1)
            hprod *= hs[-j - 1]
            vnew -= xs[-j - 1]*beta/hprod

        hnew = np.linalg.norm(vnew)
        xnew = vnew/hnew

        hs.append(hnew)
        hs.popleft()

        xs.append(xnew)
        xs.popleft() # deletes x[i - m] since the recurrence is finite

        d1 = d2
        errs[i] = measure_error(xnew, xtrue)

    return xs[-1], errs
