import numpy as np


def CreateRandomPrtf(n):   
    """Gera um array de tamanho n+1, onde suas entradas somam 1.
    Método de sorteio em um simplexo. Possui distribuição uniforme.
    int -> array"""

    aleatorio = [0]
    for _ in range(0, n):
        aleatorioi = np.random.rand()   # Gero números entre [0, 1) uniformemente
        aleatorio.append(aleatorioi)    # Os adiciono a lista.

    aleatorio.sort()                    # Ordeno-a
    aleatorio.append(1)

    pesos = np.array(aleatorio[1:]) - np.array(aleatorio[:-1])
    
    return pesos
