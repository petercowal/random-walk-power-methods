import numpy as np
from general_power_methods import momentum_general, momentum_dynamic
import matplotlib.pyplot as plt
import scipy.sparse as spspr
import scipy.io
import h5py
np.random.seed(0)

# uncomment for LaTeX style formatting
#from matplotlib import rc
#rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
#rc('text', usetex=True)

# load sparse matrix
fname = "windtunnel_evap3d" # name doesn't include .mat extension

#HDF syntax

fs = h5py.File(fname + ".mat", "r")
A = spspr.csc_matrix((fs["Problem"]["A"]["data"][:], fs["Problem"]["A"]["ir"][:], fs["Problem"]["A"]["jc"][:]))

# older .mat syntax
#A = scipy.io.loadmat(fname + ".mat")["Problem"][0][0][1]

print(A.shape)

N = A.shape[0]


print(N)

# iteration count
iter = 1000

print("finding eigenvalues")
# compute eigenvalues and plot them for reference
eigs, U = spspr.linalg.eigs(A, k = 5)
# sort them from largest to smallest magnitude
idx = np.argsort(-np.abs(eigs))

print(eigs)

eigs = eigs[idx]
U = U[:, idx]

U1 = U[:, 0]

# relative gap between magnitude of 1st and 2nd eigenvalues
# used to determine theoretical rate of convergence
spectral_gap = np.abs(eigs[0])/np.abs(eigs[1]) - 1
print(f"spectral gap = {spectral_gap}")


xinit = (np.random.rand(N)+1j*np.random.rand(N)).reshape(-1,1)
print("xinit=\n",xinit)

# apply different power methods for comparison purposes
x1, errs1 = momentum_general(A, xinit, [], iter, xtrue=U1)
print("power method err=\n",errs1[-1])

x2, errs2 = momentum_dynamic(A, xinit, [1/2, 0, 1/2], iter, xtrue=U1)
print("momentum 2 method err=\n",errs2[-1])

x3, errs3 = momentum_dynamic(A, xinit, [2/3, 0, 0, 1/3], iter, xtrue=U1)
print("momentum 3 method err=\n",errs3[-1])

x4, errs4 = momentum_dynamic(A, xinit, [3/4, 0, 0, 0, 1/4], iter, xtrue=U1)
print("momentum 4 method err=\n",errs4[-1])

cheby_params = np.array([1/2, 0, 1/2, 0, 0])
deltoid_params = np.array([2/3, 0, 0, 1/3, 0])
astroid_params = np.array([3/4, 0, 0, 0, 1/4])


x5, errs5 = momentum_dynamic(A, xinit, 0.5*cheby_params + 0.5*deltoid_params, iter, xtrue=U1)
print("hybrid method err=\n",errs4[-1])

# plot results
iters = np.arange(iter+1)
plt.subplots(figsize = (5, 5*3/4))
plt.semilogy(iters, errs1, '-', marker='o', markevery=iter//10, label = 'power method')
plt.semilogy(iters, errs2, '-', marker='x', markevery=iter//10, label = 'dynamic 2')
plt.semilogy(iters, errs3, '-', marker='^', markevery=iter//10, label = 'dynamic 3')
plt.semilogy(iters, errs4, '-', marker='s', markevery=iter//10, label = 'dynamic 4')
plt.semilogy(iters, errs5, '-', marker='*', markevery=iter//10, label = 'dynamic 2-3')

# plot theoretical asymptotic convergence as well
#asympt2 = np.pow(1+np.sqrt(2*spectral_gap), -iters)
#plt.semilogy(iters, asympt2 * errs2[-1]/asympt2[-1], '--', label = r"$O((1+\sqrt{2\varepsilon})^{-N})$")

#asympt3 = np.pow(1+np.sqrt(spectral_gap), -iters)
#plt.semilogy(iters, asympt3 * errs3[-1]/asympt3[-1], '--', label = r"$O((1+\sqrt{\varepsilon})^{-N})$")

#asympt4 = np.pow(1+np.sqrt(2*spectral_gap/3), -iters)
#plt.semilogy(iters, asympt4 * errs4[-1]/asympt4[-1], '--', label = r"$O((1+\sqrt{2\varepsilon/3})^{-N})$")

#plt.ylim(1e-10, 10)

plt.legend()
plt.xlabel("n")
plt.ylabel("relative error")
plt.tight_layout()
plt.savefig("figures/relerr_sparse_dynamic_" + fname + ".eps")
plt.show()
