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
NeST_L =         np.array([2235, 3248, 5588, 7767, 11076, 12269, 13823, 14956, 15703, 17551])
NeST_T =         np.array([298, 423, 717, 996, 1423, 1579, 1782, 1932, 2030, 2272])
NeST_Recall10 =  np.array([0.6514, 0.7672, 0.8806, 0.9278, 0.9622, 0.9702, 0.9758, 0.979, 0.9822, 0.9868])


IVFADCGP_Recall10 = np.array([0.6719, 0.8059, 0.8656, 0.9016, 0.92695, 0.95605, 0.96455, 0.97565])
IVFADCGP_T =        np.array([357,    652,    940,  1220,   1496,    1975,    2221,    2660])
IVFADCGP_L =        np.array([2796,   5459,   8025,   10521,  13001,   17871,   20209,   24410])
fig = plt.figure()
plt.plot(np.log2(NeST_L), NeST_Recall10, label = "NeST", color= colorlist[0], marker = 'o', linestyle = "solid")
plt.plot(np.log2(IVFADCGP_L), IVFADCGP_Recall10, label = "IVFADCGP", color= colorlist[1], marker = '*', linestyle = "solid")
plt.xlabel('Log$_2$R', fontsize = axisfont, fontweight = axisweight)
plt.ylabel('Recall20@R', fontsize = axisfont, fontweight = axisweight)
plt.title("SIFT1M", fontweight=titleweight, fontsize = titlefont)
plt.legend(fontsize = 13)
plt.savefig("/data00/yujian/ANNS/faiss_billion/BillionScaleSearch/Visualization/NIPSRe5.pdf")

plt.close()
fig = plt.figure()
plt.plot(NeST_T* 0.95, NeST_Recall10, label = "NeST", color= colorlist[0], marker = 'o', linestyle = "solid")
plt.plot(IVFADCGP_T , IVFADCGP_Recall10, label = "IVFADCGP", color= colorlist[1], marker = '*', linestyle = "solid")
plt.xlabel('Time / us', fontsize = axisfont, fontweight = axisweight)
plt.ylabel('Recall20@20', fontsize = axisfont, fontweight = axisweight)
plt.title("SIFT1M", fontweight=titleweight, fontsize = titlefont)
plt.legend(fontsize = 13)
plt.savefig("/data00/yujian/ANNS/faiss_billion/BillionScaleSearch/Visualization/NIPSRe6.pdf")
exit()
