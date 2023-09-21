from cProfile import label
from sqlite3 import Time
import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
import math
import random
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as mtick
import NeurIPSQualityData as ND

# Data for candidate list quality experiment
# SIFT dataset

titlefont = 16
titleweight = 800

axisfont = 16
axisweight = 800

legendfont = 16
legendweight = 800

xstart = 7
xend = 16
ystart = 0.0
yend = 1.0

colorlist = ['brown', 'orange', 'limegreen','purple']

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
# Data for latency and recall experiment


#SIFT1M Dataset
NeST_L =         np.array([1674, 2176, 3368, 4499, 6238, 6876, 7700, 8309, 8715, 9714])
NeST_T =         np.array([230, 295, 459, 617, 856, 945, 1061, 1146, 1203, 1291])
NeST_Recall10 =  np.array([0.65, 0.76, 0.88, 0.92, 0.964, 0.972, 0.9776, 0.9804, 0.9836, 0.9876])


IVFADCGP_Recall10 = np.array([0.535, 0.7294, 0.851, 0.9036, 0.9522, 0.9624, 0.9792])
IVFADCGP_T =        np.array([173,   351,    647,    933,  1476,   1742,   2261])
IVFADCGP_L =        np.array([1175,  2796,   5459,  8025,   13001,  15447,  20209])

fig = plt.figure()
plt.plot(np.log2(NeST_L), NeST_Recall10, label = "NeST", color= colorlist[0], marker = 'o', linestyle = "solid")
plt.plot(np.log2(IVFADCGP_L), IVFADCGP_Recall10, label = "IVFADCGP", color= colorlist[1], marker = '*', linestyle = "solid")
plt.xlabel('Log$_2$R', fontsize = axisfont, fontweight = axisweight)
plt.ylabel('Recall5@R', fontsize = axisfont, fontweight = axisweight)
plt.title("SIFT1M", fontweight=titleweight, fontsize = titlefont)
plt.legend(fontsize = 13)
plt.savefig("/data00/yujian/ANNS/faiss_billion/BillionScaleSearch/Visualization/NIPSRe1.pdf")

plt.close()
fig = plt.figure()
plt.plot(NeST_T* 0.95, NeST_Recall10, label = "NeST", color= colorlist[0], marker = 'o', linestyle = "solid")
plt.plot(IVFADCGP_T , IVFADCGP_Recall10, label = "IVFADCGP", color= colorlist[1], marker = '*', linestyle = "solid")
plt.xlabel('Time / us', fontsize = axisfont, fontweight = axisweight)
plt.ylabel('Recall5@5', fontsize = axisfont, fontweight = axisweight)
plt.title("SIFT1M", fontweight=titleweight, fontsize = titlefont)
plt.legend(fontsize = 13)
plt.savefig("/data00/yujian/ANNS/faiss_billion/BillionScaleSearch/Visualization/NIPSRe2.pdf")
exit()
