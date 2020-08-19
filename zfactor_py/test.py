import numpy as np

from zfactor_py.calculate_zfactor import CalculateZFactor

z = CalculateZFactor()

for Tpr in np.arange(1.0, 3.1, 0.1):
    print(f"======================@ Tpr = {Tpr} ===============================")

    for Ppr in np.arange(0, 31, 1.0):
        zANN10 = z.ANN10(Ppr, Tpr)
        zANN5 = z.ANN5(Ppr, Tpr)
        zDAK = z.DAK(Ppr, Tpr)

        print(
            f"Ppr={Ppr:<5}    "
            f"zANN10={round(zANN10, 10):<15}    "
            f"zANN5={round(zANN5, 10):<15}    "
            f"zDAK={round(zDAK, 10):<15}"
        )
