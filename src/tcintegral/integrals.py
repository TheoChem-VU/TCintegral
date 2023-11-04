import numpy as np
import matplotlib.pyplot as plt
import basis_set


bs = basis_set.BasisSet(r"C:\Users\Yuman\Desktop\pyintegral\basis_sets\cc-pvdz.1.cp2k")


for i, axis in enumerate(['x', 'y', 'z']):
    rs = np.linspace(0, 6, 100)
    Ss = []
    for r in rs:
        p1 = bs.get('C', f'1P:{axis}', (0, 0, 0))
        p2 = bs.get('C', f'1P:{axis}', (r, 0, 0))
        print(p2.center, p2.index)
        Ss.append(p1.overlap(p2))
    plt.plot(rs, Ss, label=axis)
# plt.ylim(0, 1)
plt.legend()
plt.show()
