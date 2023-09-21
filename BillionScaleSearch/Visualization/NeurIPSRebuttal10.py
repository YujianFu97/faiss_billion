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
NeST_L =         np.array([1039, 1529, 1898, 2257, 2603, 2947, 3286, 3609, 3937, 4256, 4573, 4894, 5198, 5808, 6107, 6408, 6708, 7006, 7300, 7596, 7888, 8183, 8475, 8755, 9045, 9332, 9613, 9891, 10167, 10441])
NeST_T =         np.array([149, 207, 250, 295, 340, 383, 426, 468, 509, 549, 590, 625, 661, 700, 738, 775, 813, 851, 889, 927, 964, 998, 1032, 1068, 1103, 1139, 1175, 1212, 1246, 1281])
NeST_Recall10 =  np.array([0.39, 0.53, 0.628, 0.693, 0.7395, 0.7744, 0.8067, 0.8271, 0.8454, 0.8614, 0.875, 0.8859, 0.8959, 0.9135, 0.9204, 0.9262, 0.9328, 0.9383, 0.9419, 0.9465, 0.9508, 0.9548, 0.9577, 0.9606, 0.963, 0.9646, 0.9677, 0.9705, 0.9716, 0.9729])

NeST_L_9 =   np.array([1855, 2524, 4091, 5564, 7820, 8640, 9707, 10490, 11007, 12289])
NeST_T_9 =   np.array([246, 325, 519, 704, 986, 1092, 1226, 1326, 1393, 1557])
NeST_R10_9 = np.array([0.6192, 0.7278, 0.844, 0.8949, 0.9345, 0.9421, 0.9503, 0.9542, 0.9569, 0.9614])

NeST_L_8 =   np.array([1813, 2443, 3921, 5313, 7450, 8226, 9238, 9980, 10474, 11693])
NeST_T_8 =   np.array([243, 320, 508, 687, 964, 1064, 1192, 1289, 1353, 1511])
NeST_R10_8 = np.array([0.6087, 0.713, 0.8248, 0.8728, 0.9107, 0.9179, 0.9255, 0.9293, 0.9316, 0.9358])

NeST_L_7 =   np.array([1768, 2356, 3743, 5049, 7057, 7790, 8741, 9439, 9907, 11054])
NeST_T_7 =   np.array([236, 306, 479, 642, 896, 990, 1111, 1200, 1261, 1407])
NeST_R10_7 = np.array([0.599, 0.6966, 0.8022, 0.8487, 0.8846, 0.8914, 0.898, 0.9016, 0.9038, 0.9081])

NeST_L_6 =   np.array([1722, 2267, 3559, 4780, 6656, 7344, 8232, 8866, 9323, 10399])
NeST_T_6 =   np.array([237, 308, 487, 656, 916, 1012, 1138, 1228, 1289, 1438])
NeST_R10_6 = np.array([0.5876, 0.6803, 0.7799, 0.8226, 0.8556, 0.8621, 0.8683, 0.8719, 0.8741, 0.8783])

'''
plt.figure()
plt.plot(NeST_T, NeST_Recall10, label = "k = 10", color= colorlist[0], marker = 'o', linestyle = "solid")
plt.plot(NeST_T_9, NeST_R10_9, label = "k = 9", color= colorlist[1], marker = 'o', linestyle = "solid")
plt.plot(NeST_T_8, NeST_R10_8, label = "k = 8", color= colorlist[2], marker = 'o', linestyle = "solid")
plt.plot(NeST_T_7, NeST_R10_7, label = "k = 7", color= colorlist[3], marker = 'o', linestyle = "solid")
plt.plot(NeST_T_6, NeST_R10_6, label = "k = 6", color= 'r', marker = 'o', linestyle = "solid")
plt.title("SIFT1M", fontweight=titleweight, fontsize = titlefont)
plt.legend(fontsize = 13)
plt.xlabel("T / us",fontsize = axisfont, fontweight = axisweight)
plt.ylabel("Recall10@10",fontsize = axisfont, fontweight = axisweight)
plt.savefig("/data00/yujian/ANNS/faiss_billion/BillionScaleSearch/Visualization/Tradeoff.png")
exit()
'''

IVFADCGP_L =        np.array([1175, 1727,  2796, 4390,   7676, 10521,  13001,    15447,   17871])
IVFADCGP_T =        np.array([170, 229,    344,  515,   874 , 1167,    1438,     1696,    1954])
IVFADCGP_Recall10 = np.array([0.5, 0.5937, 0.70, 0.7936, 0.88,  0.9172,   0.9398,0.9538,  0.9643])
fig = plt.figure()
plt.plot(np.log2(NeST_L), NeST_Recall10, label = "NeST", color= colorlist[0], marker = 'o', linestyle = "solid")
plt.plot(np.log2(IVFADCGP_L), IVFADCGP_Recall10, label = "IVFADCGP", color= colorlist[1], marker = '*', linestyle = "solid")
plt.xlabel('Log$_2$R', fontsize = axisfont, fontweight = axisweight)
plt.ylabel('Recall10@R', fontsize = axisfont, fontweight = axisweight)
plt.title("SIFT1M", fontweight=titleweight, fontsize = titlefont)
plt.legend(fontsize = 13)
plt.savefig("/data00/yujian/ANNS/faiss_billion/BillionScaleSearch/Visualization/NIPSRe3.pdf")

plt.close()
fig = plt.figure()
plt.plot(NeST_T, NeST_Recall10, label = "NeST", color= colorlist[0], marker = 'o', linestyle = "solid")
plt.plot(IVFADCGP_T, IVFADCGP_Recall10, label = "IVFADCGP", color= colorlist[1], marker = '*', linestyle = "solid")
plt.xlabel('Time / us', fontsize = axisfont, fontweight = axisweight)
plt.ylabel('Recall10@10', fontsize = axisfont, fontweight = axisweight)
plt.title("SIFT1M", fontweight=titleweight, fontsize = titlefont)
plt.legend(fontsize = 13)
plt.savefig("/data00/yujian/ANNS/faiss_billion/BillionScaleSearch/Visualization/NIPSRe4.pdf")
exit()


fig = plt.figure(figsize=(20, 6))
axes = fig.subplots(nrows=1, ncols=4)

plt.subplot(1, 4, 1)
# Recall1@1
# SIFT Dataset
NeST_L =        np.array([400, 978, 1145, 1246, 1442, 1854, 2491, 3517, 5768, 19000])
NeST_Recall1 =  np.array([0.1, 0.2, 0.3,  0.4,  0.5,  0.6,  0.7, 0.8,  0.9,  0.95 ])

Faiss_L =       np.array([400, 1000, 1700, 2700, 3800, 5600, 7300, 9400, 16000, 62000 ])
Faiss_Recall1 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,  0.9,  0.95 ])

FaissPLE_L =       np.array([398, 996, 1355, 2148, 3100, 5000, 6900, 8900, 15400, 61500 ])
FaissPLE_Recall1 = np.array([0.1, 0.2, 0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,   0.95 ])

IVFADCGP_L =        np.array([216, 512, 794, 1224, 1895, 3129, 4128, 5673, 9876, 45000])
IVFADCGP_Recall1 =  np.array([0.1, 0.2, 0.3, 0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  0.95 ])


''''''
axes[0].plot(np.log2(NeST_L), NeST_Recall1, label = "NeST", color= colorlist[0], marker = 'o', linestyle = "solid")
axes[0].plot(np.log2(IVFADCGP_L), IVFADCGP_Recall1, label = "IVFADCGP", color = colorlist[1], marker = '*', linestyle = "dashed")
axes[0].plot(np.log2(FaissPLE_L), FaissPLE_Recall1, label = "FaissPLE", color = colorlist[2], marker = '.', linestyle = 'dashdot')
axes[0].plot(np.log2(Faiss_L), Faiss_Recall1, label = 'Faiss', color = colorlist[3], marker = 'v', linestyle = 'dotted')

plt.xlabel('Log$_2$R', fontsize = axisfont, fontweight = axisweight)
plt.ylabel('Recall1@R', fontsize = axisfont, fontweight = axisweight)
plt.title("SIFT1B", fontweight=titleweight, fontsize = titlefont)
plt.xlim(xstart, xend)
plt.ylim(ystart, yend)

'''
axes[0, 0].plot((NeST_L), NeST_Recall1, label = "NeST", color= "black", marker = 'o', linestyle = "solid")
axes[0, 0].plot((Faiss_L), Faiss_Recall1, label = 'Faiss', color = 'black', marker = 'v', linestyle = 'dashed')
axes[0, 0].plot((IVFADCGP_L), IVFADCGP_Recall1, label = "IVFADCGP", color = 'black', marker = '*', linestyle = "dashed")
axes[0, 0].plot((FaissPLE_L), FaissPLE_Recall1, label = "FaissPLE", color = 'black', marker = '.', linestyle = 'dashed')
plt.xlim(0, 15000)
'''
#plt.legend(fontsize = 13)



# DEEP dataset
NeST_L =        np.array([365, 843, 1157, 1294, 1443, 1754, 2391, 3917, 9468, 21000 ])
NeST_Recall1 =  np.array([0.1, 0.2, 0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  0.95 ])

Faiss_L =       np.array([375, 879, 1324, 2341, 3927, 5473, 8402, 15400, 37590, 93456 ])
Faiss_Recall1 = np.array([0.1, 0.2, 0.3,  0.4,  0.5,  0.6,  0.7,  0.8,   0.9,  0.95 ])

FaissPLE_L =       np.array([330, 880, 1205, 1848, 2900, 4500, 7400, 12900, 30400, 91500 ])
FaissPLE_Recall1 = np.array([0.1, 0.2, 0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  0.95 ])

IVFADCGP_L =        np.array([197, 308, 508, 838, 1449, 2258, 4217, 8246, 22496, 60000])
IVFADCGP_Recall1 =  np.array([0.1, 0.2, 0.3, 0.4, 0.5,  0.6,  0.7,  0.8,  0.9,  0.95 ])

plt.subplot(1, 4, 2)

''''''
axes[1].plot(np.log2(NeST_L), NeST_Recall1, label = "NeST", color= colorlist[0], marker = 'o', linestyle = "solid")
axes[1].plot(np.log2(IVFADCGP_L), IVFADCGP_Recall1, label = "IVFADCGP", color = colorlist[1], marker = '*', linestyle = "dashed")
axes[1].plot(np.log2(FaissPLE_L), FaissPLE_Recall1, label = "FaissPLE", color = colorlist[2], marker = '.', linestyle = 'dashdot')
axes[1].plot(np.log2(Faiss_L), Faiss_Recall1, label = 'Faiss', color = colorlist[3], marker = 'v', linestyle = 'dotted')

plt.xlabel('Log$_2$R', fontsize = axisfont, fontweight = axisweight)
plt.ylabel('Recall1@R', fontsize = axisfont, fontweight = axisweight)
plt.title("DEEP1B", fontweight=titleweight, fontsize = titlefont)
#plt.legend(fontsize = 13)
plt.xlim(xstart, xend)
plt.ylim(ystart, yend)
'''
axes[0, 1].plot((NeST_L), NeST_Recall1, label = "NeST", color= "black", marker = 'o', linestyle = "solid")
axes[0, 1].plot((Faiss_L), Faiss_Recall1, label = 'Faiss', color = 'black', marker = 'v', linestyle = 'dashed')
axes[0, 1].plot((IVFADCGP_L), IVFADCGP_Recall1, label = "IVFADCGP", color = 'black', marker = '*', linestyle = "dashed")
axes[0, 1].plot((FaissPLE_L), FaissPLE_Recall1, label = "FaissPLE", color = 'black', marker = '.', linestyle = 'dashed')
plt.xlim(0, 15000)
'''


# Turing dataset
NeST_L =        np.array([462, 843, 1157, 1294, 1413, 1754, 2491, 3917, 7268, 18000 ])
NeST_Recall1 =  np.array([0.1, 0.2, 0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  0.95 ])

Faiss_L =           np.array([475, 979, 1324, 2441, 5327, 7873, 10702, 18400, 30590, 83456 ])
Faiss_Recall1 =     np.array([0.1, 0.2, 0.3,  0.4,  0.5,  0.6,  0.7,   0.8,  0.9,  0.95 ])

FaissPLE_L =        np.array([470, 880, 1205, 2048, 3900, 5900, 8300, 13900, 26400, 71500 ])
FaissPLE_Recall1 =  np.array([0.1, 0.2, 0.3,  0.4,  0.5,  0.6,  0.7,  0.8,   0.9,  0.95 ])

IVFADCGP_L =        np.array([183, 408, 708, 1038, 1949, 3258, 5217, 8246, 18496, 70000])
IVFADCGP_Recall1 =  np.array([0.1, 0.2, 0.3, 0.4, 0.5,  0.6,  0.7,  0.8,  0.9,  0.95 ])

plt.subplot(1, 4, 3)

''''''
axes[2].plot(np.log2(NeST_L), NeST_Recall1, label = "NeST", color= colorlist[0], marker = 'o', linestyle = "solid")
axes[2].plot(np.log2(IVFADCGP_L), IVFADCGP_Recall1, label = "IVFADCGP", color = colorlist[1], marker = '*', linestyle = "dashed")
axes[2].plot(np.log2(FaissPLE_L), FaissPLE_Recall1, label = "FaissPLE", color = colorlist[2], marker = '.', linestyle = 'dashdot')
axes[2].plot(np.log2(Faiss_L), Faiss_Recall1, label = 'Faiss', color = colorlist[3], marker = 'v', linestyle = 'dotted')


plt.xlabel('Log$_2$R', fontsize = axisfont, fontweight = axisweight)
plt.ylabel('Recall1@R', fontsize = axisfont, fontweight = axisweight)
plt.title("Turing1B", fontweight=titleweight, fontsize = titlefont)
#plt.legend(fontsize = 13)
plt.xlim(xstart, xend)
plt.ylim(ystart, yend)

'''
axes[1, 0].plot((NeST_L), NeST_Recall1, label = "NeST", color= "black", marker = 'o', linestyle = "solid")
axes[1, 0].plot((Faiss_L), Faiss_Recall1, label = 'Faiss', color = 'black', marker = 'v', linestyle = 'dashed')
axes[1, 0].plot((IVFADCGP_L), IVFADCGP_Recall1, label = "IVFADCGP", color = 'black', marker = '*', linestyle = "dashed")
axes[1, 0].plot((FaissPLE_L), FaissPLE_Recall1, label = "FaissPLE", color = 'black', marker = '.', linestyle = 'dashed')
plt.xlim(0, 15000)
'''
# SpaceV dataset
NeST_L =        np.array([379, 748, 1154, 1294, 1543, 2035, 2691, 3717, 6468, 13000 ])
NeST_Recall1 =  np.array([0.1, 0.2, 0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  0.95 ])

Faiss_L =       np.array([390, 744, 1124, 2341, 4227, 6673, 9902, 17400, 24590, 53456 ])
Faiss_Recall1 = np.array([0.1, 0.2, 0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  0.95 ])

FaissPLE_L =       np.array([394, 790, 1205, 1848, 3100, 4800, 7300, 13900, 20400, 41500 ])
FaissPLE_Recall1 = np.array([0.1, 0.2, 0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  0.95 ])

IVFADCGP_L =        np.array([137, 328, 598, 992, 1649, 2958, 4817, 7946, 14496, 50000])
IVFADCGP_Recall1 =  np.array([0.1, 0.2, 0.3, 0.4, 0.5,  0.6,  0.7,  0.8,  0.9,  0.95 ])

plt.subplot(1, 4, 4)

''''''
axes[3].plot(np.log2(NeST_L), NeST_Recall1, label = "NeST", color= colorlist[0], marker = 'o', linestyle = "solid")
axes[3].plot(np.log2(IVFADCGP_L), IVFADCGP_Recall1, label = "IVFADCGP", color = colorlist[1], marker = '*', linestyle = "dashed")
axes[3].plot(np.log2(FaissPLE_L), FaissPLE_Recall1, label = "FaissPLE", color = colorlist[2], marker = '.', linestyle = 'dashdot')
axes[3].plot(np.log2(Faiss_L), Faiss_Recall1, label = 'Faiss', color = colorlist[3], marker = 'v', linestyle = 'dotted')


plt.xlabel('Log$_2$R', fontsize = axisfont, fontweight = axisweight)
plt.ylabel('Recall1@R', fontsize = axisfont, fontweight = axisweight)
plt.title("SpaceV1B", fontweight=titleweight, fontsize = titlefont)
#plt.legend(fontsize = 13)
plt.xlim(xstart, xend)
plt.ylim(ystart, yend)

'''
axes[1, 1].plot((NeST_L), NeST_Recall1, label = "NeST", color= "black", marker = 'o', linestyle = "solid")
axes[1, 1].plot((Faiss_L), Faiss_Recall1, label = 'Faiss', color = 'black', marker = 'v', linestyle = 'dashed')
axes[1, 1].plot((IVFADCGP_L), IVFADCGP_Recall1, label = "IVFADCGP", color = 'black', marker = '*', linestyle = "dashed")
axes[1, 1].plot((FaissPLE_L), FaissPLE_Recall1, label = "FaissPLE", color = 'black', marker = '.', linestyle = 'dashed')
plt.xlim(0, 15000)
'''

lines, labels = fig.axes[-1].get_legend_handles_labels()

fig.legend(lines, labels, loc='upper center', ncol = 4, prop=dict(weight=legendweight, size = legendfont))

plt.show()
