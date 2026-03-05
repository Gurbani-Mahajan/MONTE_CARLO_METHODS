#simulating decay of Bi[213] to Bi[209] using RNG
import random
import numpy as np
import matplotlib.pyplot as plt

def decay(N_parent,p):
    cnt = 0
    for i in range(N_parent):
        if random.random() < p:
            cnt += 1
    return (cnt)

#initial number of atoms of each type
N_bi_i=10000 #Bi[213]
N_pb=0 #Pb[209]
N_tl=0 #Tl[209]
N_bi_f=0 #Bi[209]

#half-lives (in s)
t_tl=2.20*60
t_pb=3.30*60
t_bi=46*60

h=1 #increments of time in seconds (and steps for rng)

#probabilities of decay
p_tl=1-(2**(-h/t_tl))
p_pb=1-(2**(-h/t_pb))
p_bi=1-(2**(-h/t_bi))

T=20000 #total time increments (60 seconds)

#plotting points
t=np.linspace(0,T,20000)
bi_i=[]
pb=[]
tl=[]
bi_f=[]

#checking if bi decays
for j in range(T):
    bi_i.append(N_bi_i)
    tl.append(N_tl)
    pb.append(N_pb)
    bi_f.append(N_bi_f)
    decay_bi=decay(N_bi_i,p_bi) #number of decaying Bi[213] atoms
    N_bi_i-=decay_bi #subtarcting from Bi[213]
    bi_tl=0 #no. of bi[213] to tl[209] transitions
    for p in range(decay_bi):
        if random.random() < 0.0209:  # 2.09% prob of taking Bi_i ->Tl-> Pb-> Bi_f path
            bi_tl += 1
    bi_pb=decay_bi-bi_tl #no. of bi[213] to pb[209] transitions
    N_tl+=bi_tl
    N_pb+=bi_pb
    decay_tl=decay(N_tl,p_tl) #no. of tl[209] to pb[209] transitions
    N_tl-=decay_tl
    N_pb+=decay_tl
    decay_pb=decay(N_pb,p_pb) #no. of decayed Pb[209] to Bi[209]
    N_bi_f+= decay_pb
    N_pb-=decay_pb

#plotting the series
plt.plot(t,bi_i,label='Bi[213]')
plt.plot(t,tl,'r',label='Tl[209]')
plt.plot(t,pb,'y',label='Pb[209]')
plt.plot(t,bi_f,'b',label='Bi[209]')
plt.xlabel('Time')
plt.ylabel('N')
plt.legend()
plt.show()


