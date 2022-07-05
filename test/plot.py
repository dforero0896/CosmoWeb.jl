import numpy as np
import proplot as pplt

fig, ax = pplt.subplots(nrows=1, ncols=2, share=0)

k, pk0, pk2 = np.loadtxt("test.csv", delimiter=",", skiprows=1, unpack=True)

ax[0].loglog(k, pk0)
ax[1].semilogx(k, pk2)


results = np.load("tests_py.npy")
k, pk0, pk2, _, _, _, _ = np.split(results, 7, axis=1)

ax[0].loglog(k, pk0, ls='--')
ax[1].semilogx(k, pk2, ls='--')


fig.savefig("test.png", dpi=300)