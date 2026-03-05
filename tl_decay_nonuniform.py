#simulating decay of Bi[213] to Bi[209] using NON-UNIFORM RNG
import random
import numpy as np
import matplotlib.pyplot as plt

def x(z,t): #non-uniform random number in terms of inbuilt uniform random number in accordance with laws of radioactive decay
    mu=np.log(2)/t
    x0=-(1/mu)*np.log(1-z)
    return(x0)

#half-lives (in s)
t_tl=2.20*60

#initial number of atoms
N_tl=1000

#plotting points
T=np.linspace(0,N_tl,1000)
undecayed_tl=[]

#generating 1000 random times at which Bi[213] decays
t_decay=[]
for i in range(1000):
    t=x(random.random(),t_tl)
    t_decay.append(t)

t_decay=np.sort(t_decay) #sorted array of random times when Tl[209] decays

for t0 in T:
    cnt=0
    for d in t_decay:
        if d>t0:
            cnt+=1
    undecayed_tl.append(cnt) #counts no. of undecayed Bi[213] atoms at every second

plt.plot(T,undecayed_tl)
plt.minorticks_on()
plt.grid(True)
plt.grid(which="minor", linewidth=0.5)
plt.xlabel('Time')
plt.ylabel('Number of undecayed atoms')
plt.title('Radioactive Decay of 1000 Tl[209] atoms')
plt.show()

