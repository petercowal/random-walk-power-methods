import numpy as np
from general_power_methods import momentum_general, momentum_dynamic
import matplotlib.pyplot as plt
import scipy.sparse as spspr
from scipy.io import savemat
np.random.seed(0)

# uncomment for LaTeX style formatting
#from matplotlib import rc
#rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
#rc('text', usetex=True)

# generate barbell graph transition matrix
N = 1000
p = 1/250

print(1/np.sqrt(p*N))

print("generating B",end='')
rows = [N,N-1]
cols = [N-1,N]
data = [1,1]
for i in range(N):
    randos = np.random.rand(N)
    if i%1000 == 0:
        print('.',end='',flush=True)
    for j in range(N):
        if randos[j] < p:
            rows.append(i)
            cols.append(j)
            data.append(1)
for i in range(N,2*N):
    randos = np.random.rand(N)
    if i%1000 == 0:
        print('.',end='',flush=True)
    for j in range(N,2*N):
        if randos[j-N] < p:
            rows.append(i)
            cols.append(j)
            data.append(1)
B = spspr.coo_array((data, (rows, cols)), shape=(2*N, 2*N)).tocsr()

print("done!")
print("normalizing columns")
Dinv = spspr.diags(1/np.sum(B, axis=0))

A = B @ Dinv

savemat(f"barbell_N{N}_p{int(1/p)}.mat", {"A": A, "N": N, "p": p})

# iteration count
iter = 1000

print("finding eigenvalues")
# compute eigenvalues and plot them for reference
eigs, U = spspr.linalg.eigs(A, 100)
# sort them from largest to smallest magnitude
idx = np.argsort(-np.abs(eigs))
eigs = eigs[idx]
U = U[:, idx]

U1 = U[:, 0]

plt.subplots(figsize=(5*3/4, 5*3/4))

theta = np.linspace(0, 2*np.pi, 600)


plt.plot(np.cos(theta), np.sin(theta), ':', label = "$|z| = 1$")

for m in [3, 4, 5, 6]:
    curve = ((m-1)/m*np.exp(theta*1j) + 1/m*np.exp(-theta*1j*(m-1)))
    plt.plot(np.real(curve), np.imag(curve), '-', label = f"{m}-hypocycloid")

plt.plot(np.real(eigs), np.imag(eigs), 'x', label = r"$\lambda_n$")
#plt.legend()
plt.xlim(-1.2, 1.2)
plt.ylim(-1.2, 1.2)
plt.axis('square')
plt.tight_layout()
plt.savefig("figures/barbell_spectrum.eps")

xinit = (np.random.rand(2*N)+1j*np.random.rand(2*N)).reshape(-1,1)
print("xinit=\n",xinit)

# momentum parameters
beta = np.abs(eigs[1])**2/4
beta2 = 4*np.abs(eigs[1])**3/27

# relative gap between magnitude of 1st and 2nd eigenvalues
# used to determine theoretical rate of convergence
spectral_gap = np.abs(eigs[0])/np.abs(eigs[1]) - 1
print(f"spectral gap = {spectral_gap}")

# apply different power methods for comparison purposes
x1, errs1 = momentum_general(A, xinit, [], iter, xtrue=U1)
print("power method err=\n",errs1[-1])

x2, errs2 = momentum_dynamic(A, xinit, [2/3, 0, 0, 1/3], iter, xtrue=U1)
print("momentum 3 method err=\n",errs2[-1])

x3, errs3 = momentum_dynamic(A, xinit, [3/4, 0, 0, 0, 1/4], iter, xtrue=U1)
print("momentum 4 method err=\n",errs3[-1])

x4, errs4 = momentum_dynamic(A, xinit, [4/5, 0, 0, 0, 0, 1/5], iter, xtrue=U1)
print("momentum 5 method err=\n",errs4[-1])

#cheby_params = np.array([1/2, 0, 1/2, 0, 0])
#deltoid_params = np.array([2/3, 0, 0, 1/3, 0])
#astroid_params = np.array([3/4, 0, 0, 0, 1/4])


x5, errs5 = momentum_dynamic(A, xinit, [5/6, 0, 0, 0, 0, 0, 1/6], iter, xtrue=U1)
print("momentum 6 method err=\n",errs5[-1])

# plot results
iters = np.arange(iter+1)
plt.subplots(figsize = (5, 5*3/4))
plt.semilogy(iters, errs1, '-', marker='o', markevery=iter//10, label = 'power method')
plt.semilogy(iters, errs2, '-', marker='^', markevery=iter//10, label = 'dynamic 3')
plt.semilogy(iters, errs3, '-', marker='s', markevery=iter//10, label = 'dynamic 4')
plt.semilogy(iters, errs4, '-', marker='p', markevery=iter//10, label = 'dynamic 5')
plt.semilogy(iters, errs5, '-', marker='h', markevery=iter//10, label = 'dynamic 6')

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
plt.savefig("figures/relerr_barbell.eps")
plt.show()
