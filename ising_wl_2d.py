#using wang-landau mc algo to solve 2d ising model
import random
import numpy as np
import matplotlib.pyplot as plt
import math

def energy_2d(s,J):
    l=len(s)
    E_ij=0
    for i in range (l):
        for j in range (l):
            E_ij-= J * s[i,j]*(s[(i+1)%l,j] + s[i,(j+1)%l] + s[(i-1)%l,j] + s[i,(j-1)%l])
    return E_ij/2

def random_lattice(num_sites):
    s_2d = np.ones([num_sites, num_sites])
    # randomising spins so that net magnetic moment is roughly 0
    for i in range(num_sites):
        for j in range(num_sites):
            if random.random() > 0.5:
                s_2d[i, j] = 1
            else:
                s_2d[i, j] = -1
    return s_2d

def glauber(s,T):
    s_2d=s.copy()
    J=1
    E_initial=energy_2d(s_2d,J) #random starting point in energy index
    E=[E_initial]
    M=[np.sum(s_2d)]
    lattice=[s_2d] #spin configurations
    n=len(s_2d)
    steps=int(10e5)
    # generating time series
    for i in range(steps):
        r = random.randrange(n)  # random row
        c = random.randrange(n)  # random column
        s_new = s_2d[r, c] * (-1)  # flipping the spin and storing new spin
        neighbours = s_2d[r, (c - 1) % n] + s_2d[r, (c + 1) % n] + s_2d[(r - 1) % n, c] + s_2d[(r + 1) % n, c]  # modulo taken to enforce periodic boundary condition ((num_sites+1)th site=0)
        dE = 2 * J * s_2d[r, c] * neighbours #change in energy on spin flip
        E_new = E_initial + int(dE)  # new value of energy
        # deciding whether to choose the flip
        # P=1 if gj<=gi and g(Ej)/g(Ei) if gj>gi =e^ln(g(Ej))/e^ln(g(Ei))=e^(ln(g(Ej))-ln(g(Ei)))
        if dE <= 0 or np.random.rand() < np.exp(-dE / T):
            E_initial = E_new #accepting flip
            s_2d[r,c]=s_new
        M.append(np.sum(s_2d))
        E.append(E_initial)
        lattice.append(s_2d.copy())
    return np.array(M), np.array(E)


def thermalise(s,T,E,ln_g):
    s_eq=s.copy()
    n = len(s)
    E_initial = int(energy_2d(s_eq, J))
    step = 5000
    for i in range(step):
        r = random.randrange(n)  # random row
        c = random.randrange(n)  # random column
        s_new = s_eq[r, c] * (-1)  # flipping the spin and storing new spin
        neighbours = s_eq[r, (c - 1) % n] + s_eq[r, (c + 1) % n] + s_eq[(r - 1) % n, c] + s_eq[(r + 1) % n, c]
        dE = 2 * J * s_eq[r, c] * neighbours  # modulo taken to enforce periodic boundary condition ((num_sites+1)th site=0)
        E_new = E_initial + int(dE)  # new value of energy
        ind = E_new - E_min
        # checking to see if energy value is allowed, rejecting the rest of the loop if not
        if ind >= len(index_E) or index_E[ind] == -1:
            continue
        # deciding whether to choose the flip
        lng_new = ln_g[index_E[ind]]
        lng_old = ln_g[index_E[E_initial - E_min]]
        diff = lng_old - lng_new
        P = 1.0
        # P=1 if gj<=gi and g(Ej)/g(Ei) if gj>gi =e^ln(g(Ej))/e^ln(g(Ei))=e^(ln(g(Ej))-ln(g(Ei)))
        if diff < 0:
            P = np.exp(diff) * np.exp(-dE / T)
        if np.random.rand() < P:
            s_eq[r, c] = s_new  # accept the flip
            E_initial = E_new
    return s_eq


def observables(M,E,num_sites):
    #defining domains of the ferromagnetic crystal (since sum (magnetisation) over all spins is non Markovian and thus not giving required dip in spectral gap of koop eigvals at Tc
    m=M/num_sites
    e=E/num_sites
    psi=np.hstack([m,
                   m**2,
                   m**3,
                   m*e,
                   e])
    return np.array(psi)

def edmd(X1,Y1):
    n=X1.shape[0] #number of snapshots
    G=(X1.T @ X1)/n #edmd gram matrix (covariance matrix)
    A=(X1.T @ Y1)/n #edmd matrix
    #K=np.linalg.inv(F)@A
    K = np.linalg.lstsq(G + 1e-6 * np.eye(5), A, rcond=None)[0] #matrix was ill-conditioned and division by zero errors arose. had to use matrix norm
    eigvals=np.linalg.eigvals(K)
    eigvals = np.sort(np.abs(eigvals)) [::-1] # sorted eigenvalues (descending)
    return eigvals

num_sites=16 #number of lattice sites
num_steps=int(10e6) #number of steps of mc algo
J=1 #ferromagnetic
kb=1 #boltzman constant in units
s_2d=random_lattice(num_sites)

#all possible energies
n_2=num_sites*num_sites
E=np.arange(-2*n_2,2*n_2+1,4,dtype=int) #energies differ by +-4J
#removing energies not allowed due to parity symmetry
E=E.tolist()
E.pop(1)
E.pop(-2)
E=np.array(E)
E_min=E[0]
E_max=E[-1]
index_E=-np.ones(E_max-E_min+1,dtype=int) #index of allowed/quantised energies when flipping spins

for i in range(len(E)):
    index_E[E[i]-E_min]=i #looping over negative energies to start from 0

E_initial=int(energy_2d(s_2d,J))
E_plot=[E_initial]
ln_g=np.zeros(len(E)) #log of density of states function
hist=np.zeros(len(E)) #histogram for plotting
f=np.exp(1) #modification factor for W-L
ln_f=1 #ln(f)
n=len(s_2d)
n_2=len(s_2d)*len(s_2d)
print(len(hist))
ln_g_t=[] #to track g(E) as we move through the algorithm (for convergence)
flatness_t=[]

#wang-landau algorithm of random spin flipping
for i in range(num_steps):
    r=random.randrange(n) #random row
    c=random.randrange(n) #random column
    s_new = s_2d[r,c] * (-1)  # flipping the spin and storing new spin
    neighbours= s_2d[r,(c- 1) % n] + s_2d[r, (c+1)% n]+ s_2d[(r-1)%n,c]+s_2d[(r+1)%n,c]
    dE = 2 * J * s_2d[r,c] * neighbours # modulo taken to enforce periodic boundary condition ((num_sites+1)th site=0)
    E_new = E_initial + int(dE)  # new value of energy
    ind=E_new-E_min
    #checking to see if energy value is allowed, rejecting the rest of the loop if not
    if ind>=len(index_E) or index_E[ind]==-1:
        continue

    # deciding whether to choose the flip
    lng_new = ln_g[index_E[ind]]
    lng_old = ln_g[index_E[E_initial - E_min]]
    diff = lng_old - lng_new
    P=1.0
    # P=1 if gj<=gi and g(Ej)/g(Ei) if gj>gi =e^ln(g(Ej))/e^ln(g(Ei))=e^(ln(g(Ej))-ln(g(Ei)))
    if diff<0:
        P = np.exp(diff)
    if np.random.rand() < P:
        s_2d[r,c]= s_new  # accept the flip
        E_initial = E_new

    E_plot.append(E_initial)
    hist[index_E[E_initial - E_min]] += 1
    ln_g[index_E[E_initial - E_min]] += ln_f  # g->g*f so lng->lng+lnf

    # updating modification factor if histogram flattens after 1000 steps
    if (i + 1) % (1000) == 0:
        ln_g_t.append(ln_g.copy()) #after every 1000 steps
        avg = sum(hist)/len(hist)
        min_h = min(hist)
        if min_h > avg * (0.8): # measuring how flat we want the histogram to get before we change f
            flatness_t.append(min_h/avg) #tracking histogram flatness
            hist[:] = 0 #reinitialising histogram
            ln_f = ln_f / 2  # modified f (f->f*e^(-2); lnf->lnf/2)
            print('histogram details are:', avg, min_h)
            print("Modifying histogram after n= ", i + 1, " steps. New modification factor = ", np.exp(ln_f))

# normalising g with condition: g(min)=2 (2 ground states of misaligned spins) and g(max)=2 (2 highest energy states of aligned spins up or down)(g(min)+g(max)=4)(c*g(0)+c*g(N-1)=4 ->c=4/(g0+gN-1)
# normalisation constant log
print(ln_g[0])
print(ln_g[-1])
if ln_g[0] < ln_g[-1]:
    ln_c = np.log(4) - ln_g[-1] - np.log(1 + np.exp(ln_g[0] - ln_g[-1]))
else:
    ln_c = np.log(4) - ln_g[0] - np.log(1 + np.exp(ln_g[-1] - ln_g[0]))
ln_g = ln_g + ln_c
print(np.exp(ln_g[0])+np.exp(ln_g[-1]), np.exp(ln_g[0]), np.exp(ln_g[-1]))

# plotting histograms
plt.plot(E,ln_g, 'o-')
plt.ylabel('ln(g(E))')
plt.xlabel('E')
plt.title('Log of density of states for N=' + str(num_sites) + ' spin sites in 2D')
plt.grid(True)
plt.show()

#checking convergence by plotting differences
diff=[]
for g in range(len(ln_g_t)-1):
    diff.append(np.sum(np.abs(ln_g_t[g+1]-ln_g_t[g])))

plt.plot(diff)
plt.title('Convergence of log of density of states for N=' + str(num_sites) + ' spin sites in 2D')
plt.xlabel('Iteration')
plt.ylabel('L1 norm differences')
plt.grid(True)
plt.show()

#checking speed of updating the histogram
plt.plot(flatness_t)
plt.title("Histogram Flatness Over Time")
plt.ylabel("Flatness")
plt.xlabel("No. of Updates")
plt.grid()
plt.show()


#finding partition function X=sum[E](g*exp(-beta*E))
T=np.linspace(0.5,5,1000)
Z=np.zeros(len(T)) #Z/Z0 to avoid overflow
U=np.zeros(len(T)) #internal energy
U_2=np.zeros(len(T))
C_v=np.zeros(len(T))
for k in range(len(T)):
    t=T[k]
    beta=1/(kb*t)
    #taking out max to prevent overflow from numerical recipies
    #z=exp(ln_g(e)-beta*e)
    #z/z0=exp(ln_g(e)-ln_g(e_min) - (beta*e - beta*e_min))
    for i in range(len(E)):
        z=np.exp(ln_g[i]-ln_g[0]-beta*(E[i]-E_min))
        Z[k]+=z
        U[k]+=z*E[i]
        U_2[k]+=z*(E[i])**2 #sum((g*exp(-beta*e)*e)**2)
    U[k]*=1./Z[k] #U=(sum(g*exp(-beta*e)*e))/Z
    U_2[k]*=1./Z[k]
    C_v[k]=(U_2[k]-(U[k]**2))/(t**2) #Cv = d<E>/dt =(var(U[k])/(t**2)

#visualising phase transition
tc=2.269*np.ones(len(T))
plt.figure(3)
plt.plot(T,C_v,label='numerical')
plt.plot(tc,C_v,'--',label='Analytical Critical Temperature')
plt.ylabel('Cv')
plt.xlabel('T')
plt.legend()
plt.title('Variation of Specific Heat (Cv)  with T for 2D')
plt.grid(True)
plt.show()
