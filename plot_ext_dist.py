import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

Av = np.linspace(0, 5, 500)

lambda_v = {'nominal (shallowest)': 0.187,
            'steep (HP02, 5.36)': 5.36,
            'shallow (Dahlen 2012, 2.27)': 2.27,
            'kelly12 (1.0)': 1.0,
            'arp299 (Bondi+12, 0.025)': 0.025}

for label, lv in lambda_v.items():
    P = abs(1/lv) * scipy.stats.expon.pdf(Av, scale=1/lv)
    plt.plot(Av, P, label=label)

plt.xlabel(r'$A_V$ (mag)')
plt.ylabel(r'$P(A_V)$')

plt.legend(loc = "center right")
plt.tight_layout()
plt.savefig('ext_dist_options.png', dpi=150)
plt.show()
