# WANG-LANDAU MONTE CARLO SIMULATION OF ISING MODEL
This project implements the Wang–Landau Monte Carlo algorithm to compute the density of states for the Ising model.

Using the estimated density of states g(E), thermodynamic quantities such as the partition function, internal energy, and specific heat are calculated as functions of temperature.

Both 1D and 2D Ising models are implemented.

## WANG-LANDAU ALGORITHM
1. Generate a randomised spin lattice in required dimensions and find energy using formula:
E = -J*\(\Sigma \)Si*Sj
2. Generate array of all possible energies.
3. Starting with an array of g(E) initialised to zero, pick a spin from the lattice uniformly at random,flip it and compute new energy of the lattice.
4. Accept new state of lattice with probability:
P=min(g(E_old)/g(E_new),1)
5. Update g(E) with modification factor = f and accumulate histogram H(E) until it's sufficiently flat.
6. Once the threshold is reached, update f to (\sqrt \)(f) and reset the histogram.
7. Compute thermodynamic quantities from g(E)

## CODE STRUCTURE

ising_wl_1d.py  
→ Wang–Landau simulation for the 1D Ising model

ising_wl_2d.py  
→ Wang–Landau simulation for the 2D Ising model

Outputs:
- density of states ln g(E) 
- partition function Z(T)
- specific heat C_v(T)

## References

Wang, F. and Landau, D.P., 2001. Efficient, multiple-range random walk algorithm to calculate the density of states. Physical Review Letters.

Newman, M. Computational Physics, 1st edition, 2013.
