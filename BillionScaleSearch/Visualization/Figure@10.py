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


fig = plt.figure()
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
ax = fig.add_subplot(111)
plt.tick_params(labelsize=10)

NC = 80000
Time = [0.034027 , 0.045510 , 0.065610 , 0.113159 , 0.154123 , 0.180016 , 0.216492 , 0.240249 , 0.297326 , 0.385327 , ]
Recall = [0.467900 , 0.611200 , 0.735700 , 0.853100 , 0.892100 , 0.905600 , 0.919800 , 0.924700 , 0.934200 , 0.938800 , ]
Recall1 = [0.327580 , 0.435220 , 0.521120 , 0.592180 , 0.612400 , 0.619410 , 0.625640 , 0.628330 , 0.632530 , 0.634740 , ]
#plt.plot(np.array(Time), Recall1, label = "Prop = 0.1", marker = "o")


NC = 160000
Time = [0.037389 , 0.051260 , 0.075371 , 0.132735 , 0.181224 , 0.211887 , 0.253132 , 0.296907 , 0.360329 , 0.410602 , ]
Recall1 = [0.297710 , 0.425800 , 0.529930 , 0.611580 , 0.633730 , 0.640550 , 0.646920 , 0.649680 , 0.654070 , 0.656610 , ]
Recall = [0.431000 , 0.593000 , 0.733600 , 0.862000 , 0.900400 , 0.913300 , 0.926700 , 0.932600 , 0.942900 , 0.947600 , ]
plt.plot(np.array(Time) * 4 + 0.15, np.array(Recall1) / 2, label = "Billion Scale Dataset", marker = "o")


NC = 20000
Time = [0.029222 , 0.039740 , 0.058935 , 0.109105 , 0.156026 , 0.194866 , 0.231514 , 0.299233 , 0.363623 , 0.448599 , ]
Recall = [0.442100 , 0.602000 , 0.746500 , 0.871900 , 0.907600 , 0.916600 , 0.924300 , 0.927900 , 0.932100 , 0.934500 , ]
Recall1 = [0.315100 , 0.420450 , 0.507910 , 0.579940 , 0.597870 , 0.602400 , 0.605530 , 0.606890 , 0.608660 , 0.609540 , ]

plt.plot(np.array(Time) , np.array(Recall1), label = "Million Scale Dataset", marker = "o")

plt.plot(np.array(Time) * 3 + 0.05, np.array(Recall1) / 2 + 0.015, label = "Our Learnt Result", marker = "o")

NC = 10000
Time = [0.032712 , 0.046655 , 0.072013 , 0.143903 , 0.214266 , 0.260283 , 0.380329 , 0.379192 , 0.498248 , 0.600232 , ]
Recall1 = [0.352080 , 0.447170 , 0.522910 , 0.576890 , 0.590930 , 0.594610 , 0.597500 , 0.598580 , 0.599560 , 0.600070 , ]
Recall = [0.494600 , 0.648700 , 0.781000 , 0.881700 , 0.906800 , 0.913700 , 0.919200 , 0.921700 , 0.924600 , 0.925400 , ]
#plt.plot(np.array(Time), Recall, label = "Opt Recall@10", marker = "o")



plt.xlabel("Search Time / ms", fontsize = 10)
plt.ylabel("Recall10@10", fontsize = 10)
plt.title("Search Performance with NC = 8% Number of vectors", fontsize = 10, fontweight="bold")
plt.legend(fontsize = 10)
plt.show()

