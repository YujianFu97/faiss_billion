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
NeST_L =         np.array([3071, 4860, 8982, 12815, 18579, 20633, 23319, 25250, 26538, 29681])
NeST_T =         np.array([453, 690, 1235, 1755, 2539, 2824, 3186, 3453, 3624, 4051])
NeST_Recall10 =  np.array([0.522, 0.639, 0.783, 0.854, 0.914, 0.928, 0.942, 0.949, 0.954, 0.963])


IVFADCGP_Recall10 = np.array([0.58])
IVFADCGP_T =        np.array([])
IVFADCGP_L =        np.array([])
fig = plt.figure()
plt.plot(np.log2(NeST_L), NeST_Recall10, label = "NeST", color= colorlist[0], marker = 'o', linestyle = "solid")
plt.plot(np.log2(IVFADCGP_L), IVFADCGP_Recall10, label = "IVFADCGP", color= colorlist[1], marker = '*', linestyle = "solid")
plt.xlabel('Log$_2$R', fontsize = axisfont, fontweight = axisweight)
plt.ylabel('Recall50@R', fontsize = axisfont, fontweight = axisweight)
plt.title("SIFT1M", fontweight=titleweight, fontsize = titlefont)
plt.legend(fontsize = 13)
plt.savefig("/data00/yujian/ANNS/faiss_billion/BillionScaleSearch/Visualization/NIPSRe3.png")

plt.close()
fig = plt.figure()
plt.plot(NeST_T* 0.95, NeST_Recall10, label = "NeST", color= colorlist[0], marker = 'o', linestyle = "solid")
plt.plot(IVFADCGP_T , IVFADCGP_Recall10, label = "IVFADCGP", color= colorlist[1], marker = '*', linestyle = "solid")
plt.xlabel('T/us', fontsize = axisfont, fontweight = axisweight)
plt.ylabel('Recall50@50', fontsize = axisfont, fontweight = axisweight)
plt.title("SIFT1M", fontweight=titleweight, fontsize = titlefont)
plt.legend(fontsize = 13)
plt.savefig("/data00/yujian/ANNS/faiss_billion/BillionScaleSearch/Visualization/NIPSRe4.png")
exit()
