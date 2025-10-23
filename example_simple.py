import numpy as np
from general_power_methods import momentum_general, momentum_dynamic
import matplotlib.pyplot as plt
np.random.seed(0)

# uncomment for LaTeX style formatting
#from matplotlib import rc
#rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
#rc('text', usetex=True)

# simple 4x4 example with eigenvalues that lie within a deltoid
N = 4
A = np.array([[1.01, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1/2], [0, 0, 1/2, 0]])
U1 = np.zeros(N)
U1[0] = 1

# iteration count
iter = 300

# relative gap between magnitude of 1st and 2nd eigenvalues
# used to determine theoretical rate of convergence
spectral_gap = 0.01

print("A=\n",A)

xinit = (np.random.rand(N)+1j*np.random.rand(N)).reshape(-1,1)
print("xinit=\n",xinit)

# momentum parameters
beta2 = 1/4
beta3 = 4/27
beta4 = 27/256
beta5 = (4**4)/(5**5)


# apply different power methods for comparison purposes
x1, errs1 = momentum_general(A, xinit, [], iter, p0 = 1, xtrue=U1)
print("power method err=\n",errs1[-1])

x2, errs2 = momentum_general(A, xinit, [0, beta2], iter, p0 = 1/2, xtrue=U1)
print("momentum 2 method err=\n",errs2[-1])

x3, errs3 = momentum_general(A, xinit, [0, 0, beta3], iter, p0 = 2/3, xtrue=U1)
print("momentum 3 method err=\n",errs3[-1])

x4, errs4 = momentum_general(A, xinit, [0, 0, 0, beta4], iter, p0 = 3/4, xtrue=U1)
print("momentum 4 method err=\n",errs4[-1])

x5, errs5 = momentum_general(A, xinit, [0, 0, 0, 0, beta5], iter, p0 = 4/5, xtrue=U1)
print("momentum 5 method err=\n",errs5[-1])

# plot results
iters = np.arange(iter+1)
plt.subplots(figsize = (5, 5*3/4))
plt.semilogy(iters, errs1, '-', marker='o', markevery=iter//10, label = 'power method')
plt.semilogy(iters, errs2, '-', marker='|', markevery=iter//10, label = 'momentum 2')
plt.semilogy(iters, errs3, '-', marker='^', markevery=iter//10, label = 'momentum 3')
plt.semilogy(iters, errs4, '-', marker='s', markevery=iter//10, label = 'momentum 4')
plt.semilogy(iters, errs5, '-', marker='p', markevery=iter//10, label = 'momentum 5')
# plot theoretical asymptotic convergence as well
#asympt2 = np.pow(1+np.sqrt(spectral_gap), -iters)
#plt.semilogy(iters, asympt2 * errs2[-1]/asympt2[-1], '--', label = r"$O((1+\sqrt{\varepsilon})^{-N})$")

asympt4 = np.pow(1+np.sqrt(2*spectral_gap/3), -iters)
plt.semilogy(iters, asympt4 * errs4[-1]/asympt4[-1], '--', label = r"$O((1+\sqrt{2\varepsilon/3})^{-N})$")

asympt5 = np.pow(1+np.sqrt(2*spectral_gap/4), -iters)
plt.semilogy(iters, asympt5 * errs5[-1]/asympt5[-1], '--', label = r"$O((1+\sqrt{2\varepsilon/4})^{-N})$")

#plt.ylim(1e-10, 10)

plt.legend()
plt.xlabel("n")
plt.ylabel("relative error")
plt.tight_layout()
plt.savefig("figures/relerr_simple.eps")
plt.show()
