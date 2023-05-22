from cProfile import label
from sqlite3 import Time
from turtle import color
import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
import math
import random
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as mtick

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
colorlist = ['brown', 'orange', 'limegreen','purple']

titlefont = 16
titleweight = 800

axisfont = 16
axisweight = 800

legendfont = 16
legendweight = 800

RatioDEEP1B = [0.572718, 0.732977, 0.800465, 0.836334, 0.859874, 0.875952, 0.887796, 0.896176, 0.902973, 0.909228]
RatioSIFT1B = [0.650567, 0.816197, 0.876782, 0.907396, 0.925399, 0.936577, 0.944674, 0.951013, 0.956016, 0.959658]

x = range(1, 11)

plt.plot(x, RatioSIFT1B, label = "SIFT1B", marker = 'v', color = colorlist[0], linestyle = "solid")
plt.plot(x, RatioDEEP1B, label = "DEEP1B", marker = 'o', color = colorlist[1], linestyle = "solid")
plt.xticks(np.arange(0, 11, step=1))
plt.xlabel("$K$", fontsize = axisfont, fontweight = axisweight)
plt.ylabel("Proportion of candidates \n in neighboring partitions", fontsize = axisfont, fontweight = axisweight)
plt.xlim(0, 11)
plt.ylim(0.5, )
plt.legend(prop=dict(weight=legendweight, size = legendfont))
plt.show()
