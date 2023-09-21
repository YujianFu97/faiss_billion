import math
import numpy as np


def ip(x, y):
    sum = 0
    for i in range(x.shape[0]):
        sum += x[i] * y[i]
    return sum

def pointvector(x, y):
    vec = []
    for i in range(x.shape[0]):
        vec.append(y[i] - x[i])
    return np.array(vec)

def l2(x, y):
    sum = 0
    for i in range(x.shape[0]):
        sum += (x[i] - y[i]) * (x[i] - y[i])
    return sum

A = np.array([1, 0, 0, 1, 9])
B = np.array([4, 0, 3, 2, 18])
C = np.array([5, 2, 0, 3, 13])
D = np.array([1, 1, 1, 4, 10])

DA = pointvector(D, A)
BC = pointvector(B, C)
AC = pointvector(A, C)
AB = pointvector(A, B)

dAB = l2(A, B)
dCD = l2(C, D)
dBC = l2(B, C)

dAD = l2(A, D)
dAC = l2(A, C)
dBD = l2(B, D)

alpha = (ip(DA, AC) * ip(AC, BC) - ip(DA , BC)* ip(AC, AC)) / (ip(AB, BC) * ip(AC, AC) - ip(AB, AC) * ip(AC, BC))
beta =  (ip(DA, BC) * ip(AB, AB) - ip(DA, AB) * ip(AB, BC)) / (ip(AC, AB) * ip(AB,BC) - ip(AB, AB) * ip(AC, BC))

print(alpha, beta)

AO = alpha * AB + beta * AC
DO = DA + AO
print(ip(AO, DO))

top_alpha =    ip(DA, AC) * ip(AC, BC) - ip(DA , BC)* ip(AC, AC)
top_l2_alpha = 0.5 * (dCD - dAC-dAD) * 0.5 * (dAC + dBC - dAB) - 0.5 * (dAB - dAC - dBD + dCD) * (dAC)
below_alpha =  ip(AB, BC) * ip(AC, AC) - ip(AB, AC) * ip(AC, BC)
below_l2_alpha = 0.5 * (dAC - dAB - dBC) * (dAC) - 0.5 * (dAB + dAC- dBC) * 0.5 * (dAC + dBC - dAB)
top_beta =     ip(DA, BC) * ip(AB, AB) - ip(DA, AB) * ip(AB, BC)
top_l2_beta =  0.5 * (dAB - dAC - dBD + dCD) * dAB - 0.5 * (dBD - dAB - dAD) * 0.5 * (dAC - dAB - dBC)
below_beta =   ip(AC, AB) * ip(AB,BC) - ip(AB, AB) * ip(AC, BC)
below_l2_beta = 0.5 * (dAB + dAC - dBC) * 0.5 * (dAC - dAB - dBC) - dAB * 0.5 * (dAC + dBC - dAB)

alpha_l2 =  (0.5 * (dCD - dAC- dAD) * 0.5 * (dAC + dBC - dAB) - 0.5 * (dAB - dAC - dBD + dCD) * (dAC)) / (0.5 * (dAC - dAB - dBC) * (dAC) - 0.5 * (dAB + dAC- dBC) * 0.5 * (dAC + dBC - dAB))
beta_l2 =  (0.5 * (dAB - dAC - dBD + dCD) * dAB - 0.5 * (dBD - dAB - dAD) * 0.5 * (dAC - dAB - dBC)) / (0.5 * (dAB + dAC - dBC) * 0.5 * (dAC - dAB - dBC) - dAB * 0.5 * (dAC + dBC - dAB))

top_l2_alpha_2 = 0.25 * (-dAC*dAD+dBC*dCD-dAC*dBC-dAD*dBC-dAB*dCD+dAB*dAD-dAB*dAC+dAC*dAC+2*dAC*dBD-dAC*dCD)
top_l2_alpha_2 = 0.25 * (dAD*(-dAC-dBC+dAB)+dBD*(2*dAC)+dCD*(dBC-dAC-dAB)-dAC*dBC-dAB*dAC+dAC*dAC)
below_l2_alpha_2 = 0.25 * (dAC*dAC+dAB*dAB+dBC*dBC-2*dAC*dBC-2*dAB*dBC-2*dAB*dAC)
top_l2_beta_2 = 0.25*(dAB*dAB-dAC*dAB-dAB*dBD+2*dAB*dCD-dAC*dBD+dBC*dBD-dAB*dBC+dAC*dAD-dAB*dAD-dAD*dBC)
top_l2_beta_2 = 0.25 * (dAD*(dAC-dAB-dBC)+dBD*(dBC-dAC-dAB)+dCD*(2*dAB)+dAB*dAB-dAC*dAB-dAB*dBC)
below_l2_beta_2 = 0.25 * (dAC*dAC+dBC*dBC+dAB*dAB-2*dAB*dAC-2*dAB*dBC-2*dAC*dBC)

print(below_l2_beta, below_l2_beta_2)
AO = alpha_l2 * AB + beta_l2 * AC
DO = DA + AO

print(ip(BC, DO))
