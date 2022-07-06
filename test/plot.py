import numpy as np
import proplot as pplt

fig, ax = pplt.subplots(nrows=1, ncols=2, share=0)

k, pk0, pk2, nmodes = np.loadtxt("test.csv", delimiter=",", skiprows=1, unpack=True)

ax[0].semilogx(k, k*pk0)
ax[1].semilogx(k, pk2)


results = np.load("tests_py.npy")
k, pk0, pk2, _, _, n_modes= np.split(results, 6, axis=1)
mask = k < 1.8
k = k[mask]
pk0 = pk0[mask]
pk2 = pk2[mask]
ax[0].semilogx(k, k*pk0, ls='--')
ax[1].semilogx(k, pk2, ls='--')

print(n_modes)
print(pk0)
fig.savefig("test.png", dpi=300)