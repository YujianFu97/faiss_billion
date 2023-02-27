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


Data1 = ['Numcluster: 200 Cluster Batch: 20 Visited num of GT in top 10 : 5.2495 Candidate List Size: 500.88 / 100000 Target recall: 3.5 Search Time: 0.174962 Cen Search Time: 0.150143 Table time: 0.0088403 Vec Search Time: 0.0159782 with repeat times: 0 / 3',
'Numcluster: 180 Cluster Batch: 20 Visited num of GT in top 10 : 5.2366 Candidate List Size: 451.086 / 100000 Target recall: 3.5 Search Time: 0.153972 Cen Search Time: 0.135491 Table time: 0.0086772 Vec Search Time: 0.0098037 with repeat times: 1 / 3',
'Numcluster: 160 Cluster Batch: 20 Visited num of GT in top 10 : 5.2184 Candidate List Size: 401.121 / 100000 Target recall: 3.5 Search Time: 0.135809 Cen Search Time: 0.118132 Table time: 0.0086434 Vec Search Time: 0.0090329 with repeat times: 0 / 3',
'Numcluster: 140 Cluster Batch: 20 Visited num of GT in top 10 : 5.1953 Candidate List Size: 351.087 / 100000 Target recall: 3.5 Search Time: 0.130409 Cen Search Time: 0.113482 Table time: 0.00874 Vec Search Time: 0.0081869 with repeat times: 1 / 3',
'Numcluster: 120 Cluster Batch: 20 Visited num of GT in top 10 : 5.1606 Candidate List Size: 301.221 / 100000 Target recall: 3.5 Search Time: 0.110218 Cen Search Time: 0.0943104 Table time: 0.0086027 Vec Search Time: 0.0073053 with repeat times: 0 / 3',
'Numcluster: 100 Cluster Batch: 20 Visited num of GT in top 10 : 5.1115 Candidate List Size: 251.186 / 100000 Target recall: 3.5 Search Time: 0.101704 Cen Search Time: 0.0865204 Table time: 0.0086234 Vec Search Time: 0.0065605 with repeat times: 0 / 3',
'Numcluster: 80 Cluster Batch: 20 Visited num of GT in top 10 : 5.0369 Candidate List Size: 201.13 / 100000 Target recall: 3.5 Search Time: 0.0847141 Cen Search Time: 0.0705376 Table time: 0.0085904 Vec Search Time: 0.0055861 with repeat times: 0 / 3',
'Numcluster: 60 Cluster Batch: 20 Visited num of GT in top 10 : 4.9175 Candidate List Size: 150.977 / 100000 Target recall: 3.5 Search Time: 0.0727402 Cen Search Time: 0.0594449 Table time: 0.0086235 Vec Search Time: 0.0046718 with repeat times: 0 / 3',
'Numcluster: 40 Cluster Batch: 20 Visited num of GT in top 10 : 4.6751 Candidate List Size: 100.762 / 100000 Target recall: 3.5 Search Time: 0.0587058 Cen Search Time: 0.0462497 Table time: 0.0086484 Vec Search Time: 0.0038077 with repeat times: 0 / 3',
'Numcluster: 20 Cluster Batch: 20 Visited num of GT in top 10 : 4.0443 Candidate List Size: 50.5547 / 100000 Target recall: 3.5 Search Time: 0.0402048 Cen Search Time: 0.028782 Table time: 0.0086406 Vec Search Time: 0.0027822 with repeat times: 0 / 3',
'Numcluster: 1 Cluster Batch: 20 Visited num of GT in top 10 : 0.4633 Candidate List Size: 2.4973 / 100000 Target recall: 3.5 Search Time: 0.01488 Cen Search Time: 0.0058707 Table time: 0.0086233 Vec Search Time: 0.000386 with repeat times: 0 / 3',
'Numcluster: 19 Cluster Batch: 2 Visited num of GT in top 10 : 3.9844 Candidate List Size: 48.051 / 100000 Target recall: 3.5 Search Time: 0.0390523 Cen Search Time: 0.0277434 Table time: 0.0085825 Vec Search Time: 0.0027264 with repeat times: 3 / 3',
'Numcluster: 17 Cluster Batch: 2 Visited num of GT in top 10 : 3.853 Candidate List Size: 43.0098 / 100000 Target recall: 3.5 Search Time: 0.0368413 Cen Search Time: 0.025662 Table time: 0.0085793 Vec Search Time: 0.0026 with repeat times: 0 / 3',
'Numcluster: 15 Cluster Batch: 2 Visited num of GT in top 10 : 3.6981 Candidate List Size: 37.9691 / 100000 Target recall: 3.5 Search Time: 0.0351239 Cen Search Time: 0.024033 Table time: 0.0086251 Vec Search Time: 0.0024658 with repeat times: 0 / 3',
'Numcluster: 13 Cluster Batch: 2 Visited num of GT in top 10 : 3.5115 Candidate List Size: 32.9523 / 100000 Target recall: 3.5 Search Time: 0.0325451 Cen Search Time: 0.0216827 Table time: 0.0085496 Vec Search Time: 0.0023128 with repeat times: 0 / 3',
'Numcluster: 11 Cluster Batch: 2 Visited num of GT in top 10 : 3.2774 Candidate List Size: 27.8795 / 100000 Target recall: 3.5 Search Time: 0.0304979 Cen Search Time: 0.0197396 Table time: 0.0086072 Vec Search Time: 0.0021511 with repeat times: 0 / 3',
'Numcluster: 12 Cluster Batch: 1 Visited num of GT in top 10 : 3.4036 Candidate List Size: 30.4156 / 100000 Target recall: 3.5 Search Time: 0.0317459 Cen Search Time: 0.0208789 Table time: 0.0086275 Vec Search Time: 0.0022395 with repeat times: 3 / 3',
]


Data2 = ['Numcluster: 200 Cluster Batch: 20 Visited num of GT in top 10 : 5.0063 Candidate List Size: 1002.73 / 200000 Target recall: 3.5 Search Time: 0.162398 Cen Search Time: 0.137372 Table time: 0.0085767 Vec Search Time: 0.01645 with repeat times: 0 / 3',
'Numcluster: 180 Cluster Batch: 20 Visited num of GT in top 10 : 4.9994 Candidate List Size: 902.826 / 200000 Target recall: 3.5 Search Time: 0.150656 Cen Search Time: 0.127057 Table time: 0.0085434 Vec Search Time: 0.0150557 with repeat times: 0 / 3',
'Numcluster: 160 Cluster Batch: 20 Visited num of GT in top 10 : 4.9865 Candidate List Size: 802.816 / 200000 Target recall: 3.5 Search Time: 0.138995 Cen Search Time: 0.116737 Table time: 0.0085756 Vec Search Time: 0.0136826 with repeat times: 0 / 3',
'Numcluster: 140 Cluster Batch: 20 Visited num of GT in top 10 : 4.9738 Candidate List Size: 702.76 / 200000 Target recall: 3.5 Search Time: 0.126386 Cen Search Time: 0.105436 Table time: 0.0085682 Vec Search Time: 0.0123814 with repeat times: 0 / 3',
'Numcluster: 120 Cluster Batch: 20 Visited num of GT in top 10 : 4.9539 Candidate List Size: 602.821 / 200000 Target recall: 3.5 Search Time: 0.113845 Cen Search Time: 0.0943585 Table time: 0.0085616 Vec Search Time: 0.0109249 with repeat times: 0 / 3',
'Numcluster: 100 Cluster Batch: 20 Visited num of GT in top 10 : 4.9246 Candidate List Size: 502.74 / 200000 Target recall: 3.5 Search Time: 0.101676 Cen Search Time: 0.083627 Table time: 0.0085171 Vec Search Time: 0.0095319 with repeat times: 0 / 3',
'Numcluster: 80 Cluster Batch: 20 Visited num of GT in top 10 : 4.8753 Candidate List Size: 402.546 / 200000 Target recall: 3.5 Search Time: 0.0879179 Cen Search Time: 0.0711765 Table time: 0.0085724 Vec Search Time: 0.008169 with repeat times: 0 / 3',
'Numcluster: 60 Cluster Batch: 20 Visited num of GT in top 10 : 4.7945 Candidate List Size: 302.303 / 200000 Target recall: 3.5 Search Time: 0.0739026 Cen Search Time: 0.0585192 Table time: 0.0085518 Vec Search Time: 0.0068316 with repeat times: 0 / 3',
'Numcluster: 40 Cluster Batch: 20 Visited num of GT in top 10 : 4.6336 Candidate List Size: 201.727 / 200000 Target recall: 3.5 Search Time: 0.0583795 Cen Search Time: 0.0446297 Table time: 0.0085731 Vec Search Time: 0.0051767 with repeat times: 0 / 3',
'Numcluster: 20 Cluster Batch: 20 Visited num of GT in top 10 : 4.1794 Candidate List Size: 101.19 / 200000 Target recall: 3.5 Search Time: 0.0408605 Cen Search Time: 0.0285909 Table time: 0.0086004 Vec Search Time: 0.0036692 with repeat times: 0 / 3',
'Numcluster: 1 Cluster Batch: 20 Visited num of GT in top 10 : 0.6282 Candidate List Size: 5.0421 / 200000 Target recall: 3.5 Search Time: 0.0150298 Cen Search Time: 0.0058333 Table time: 0.008568 Vec Search Time: 0.0006285 with repeat times: 0 / 3',
'Numcluster: 19 Cluster Batch: 2 Visited num of GT in top 10 : 4.1322 Candidate List Size: 96.1681 / 200000 Target recall: 3.5 Search Time: 0.0394782 Cen Search Time: 0.0273273 Table time: 0.0085577 Vec Search Time: 0.0035932 with repeat times: 3 / 3',
'Numcluster: 17 Cluster Batch: 2 Visited num of GT in top 10 : 4.0291 Candidate List Size: 86.088 / 200000 Target recall: 3.5 Search Time: 0.0373461 Cen Search Time: 0.0254067 Table time: 0.0085159 Vec Search Time: 0.0034235 with repeat times: 0 / 3',
'Numcluster: 15 Cluster Batch: 2 Visited num of GT in top 10 : 3.9048 Candidate List Size: 75.96 / 200000 Target recall: 3.5 Search Time: 0.0353715 Cen Search Time: 0.0235175 Table time: 0.0086251 Vec Search Time: 0.0032289 with repeat times: 0 / 3',
'Numcluster: 13 Cluster Batch: 2 Visited num of GT in top 10 : 3.7569 Candidate List Size: 65.9164 / 200000 Target recall: 3.5 Search Time: 0.0332678 Cen Search Time: 0.0216534 Table time: 0.0085584 Vec Search Time: 0.003056 with repeat times: 0 / 3',
'Numcluster: 11 Cluster Batch: 2 Visited num of GT in top 10 : 3.5691 Candidate List Size: 55.7788 / 200000 Target recall: 3.5 Search Time: 0.0309445 Cen Search Time: 0.0195796 Table time: 0.0085169 Vec Search Time: 0.002848 with repeat times: 0 / 3',
'Numcluster: 9 Cluster Batch: 2 Visited num of GT in top 10 : 3.3159 Candidate List Size: 45.6439 / 200000 Target recall: 3.5 Search Time: 0.0282969 Cen Search Time: 0.0171199 Table time: 0.0085723 Vec Search Time: 0.0026047 with repeat times: 0 / 3',
'Numcluster: 10 Cluster Batch: 1 Visited num of GT in top 10 : 3.448 Candidate List Size: 50.7378 / 200000 Target recall: 3.5 Search Time: 0.0294397 Cen Search Time: 0.0181505 Table time: 0.0085648 Vec Search Time: 0.0027244 with repeat times: 3 / 3',
]

Data3 = ['Numcluster: 200 Cluster Batch: 20 Visited num of GT in top 10 : 4.8612 Candidate List Size: 1505.7 / 300000 Target recall: 3.5 Search Time: 0.167814 Cen Search Time: 0.137372 Table time: 0.0085966 Vec Search Time: 0.0218462 with repeat times: 0 / 3',
'Numcluster: 180 Cluster Batch: 20 Visited num of GT in top 10 : 4.8548 Candidate List Size: 1355.65 / 300000 Target recall: 3.5 Search Time: 0.15528 Cen Search Time: 0.126968 Table time: 0.0086509 Vec Search Time: 0.0196609 with repeat times: 0 / 3',
'Numcluster: 160 Cluster Batch: 20 Visited num of GT in top 10 : 4.8462 Candidate List Size: 1205.4 / 300000 Target recall: 3.5 Search Time: 0.142941 Cen Search Time: 0.116466 Table time: 0.008626 Vec Search Time: 0.0178492 with repeat times: 0 / 3',
'Numcluster: 140 Cluster Batch: 20 Visited num of GT in top 10 : 4.8366 Candidate List Size: 1055.16 / 300000 Target recall: 3.5 Search Time: 0.130041 Cen Search Time: 0.105366 Table time: 0.0086377 Vec Search Time: 0.0160375 with repeat times: 0 / 3',
'Numcluster: 120 Cluster Batch: 20 Visited num of GT in top 10 : 4.8199 Candidate List Size: 905.042 / 300000 Target recall: 3.5 Search Time: 0.116883 Cen Search Time: 0.0941507 Table time: 0.0086435 Vec Search Time: 0.0140884 with repeat times: 0 / 3',
'Numcluster: 100 Cluster Batch: 20 Visited num of GT in top 10 : 4.7978 Candidate List Size: 754.786 / 300000 Target recall: 3.5 Search Time: 0.103536 Cen Search Time: 0.0826652 Table time: 0.0085861 Vec Search Time: 0.0122848 with repeat times: 0 / 3',
'Numcluster: 80 Cluster Batch: 20 Visited num of GT in top 10 : 4.7584 Candidate List Size: 604.322 / 300000 Target recall: 3.5 Search Time: 0.0896821 Cen Search Time: 0.0707583 Table time: 0.0086069 Vec Search Time: 0.0103169 with repeat times: 0 / 3',
'Numcluster: 60 Cluster Batch: 20 Visited num of GT in top 10 : 4.6953 Candidate List Size: 453.881 / 300000 Target recall: 3.5 Search Time: 0.0754736 Cen Search Time: 0.058384 Table time: 0.0086067 Vec Search Time: 0.0084829 with repeat times: 0 / 3',
'Numcluster: 40 Cluster Batch: 20 Visited num of GT in top 10 : 4.558 Candidate List Size: 302.957 / 300000 Target recall: 3.5 Search Time: 0.0595448 Cen Search Time: 0.044492 Table time: 0.0086369 Vec Search Time: 0.0064159 with repeat times: 0 / 3',
'Numcluster: 20 Cluster Batch: 20 Visited num of GT in top 10 : 4.1697 Candidate List Size: 151.979 / 300000 Target recall: 3.5 Search Time: 0.0413441 Cen Search Time: 0.0283421 Table time: 0.0086574 Vec Search Time: 0.0043446 with repeat times: 0 / 3',
'Numcluster: 1 Cluster Batch: 20 Visited num of GT in top 10 : 0.7166 Candidate List Size: 7.5958 / 300000 Target recall: 3.5 Search Time: 0.0155697 Cen Search Time: 0.0061344 Table time: 0.0086042 Vec Search Time: 0.0008311 with repeat times: 0 / 3',
'Numcluster: 19 Cluster Batch: 2 Visited num of GT in top 10 : 4.1326 Candidate List Size: 144.443 / 300000 Target recall: 3.5 Search Time: 0.046071 Cen Search Time: 0.0317397 Table time: 0.0096439 Vec Search Time: 0.0046874 with repeat times: 3 / 3',
'Numcluster: 17 Cluster Batch: 2 Visited num of GT in top 10 : 4.0488 Candidate List Size: 129.275 / 300000 Target recall: 3.5 Search Time: 0.0424295 Cen Search Time: 0.0292506 Table time: 0.008828 Vec Search Time: 0.0043509 with repeat times: 0 / 3',
'Numcluster: 15 Cluster Batch: 2 Visited num of GT in top 10 : 3.9356 Candidate List Size: 114.086 / 300000 Target recall: 3.5 Search Time: 0.0368764 Cen Search Time: 0.0244364 Table time: 0.0086278 Vec Search Time: 0.0038122 with repeat times: 0 / 3',
'Numcluster: 13 Cluster Batch: 2 Visited num of GT in top 10 : 3.8113 Candidate List Size: 99.0108 / 300000 Target recall: 3.5 Search Time: 0.0345081 Cen Search Time: 0.0223236 Table time: 0.0086228 Vec Search Time: 0.0035617 with repeat times: 0 / 3',
'Numcluster: 11 Cluster Batch: 2 Visited num of GT in top 10 : 3.6497 Candidate List Size: 83.762 / 300000 Target recall: 3.5 Search Time: 0.0322501 Cen Search Time: 0.0200679 Table time: 0.0088419 Vec Search Time: 0.0033403 with repeat times: 0 / 3',
'Numcluster: 9 Cluster Batch: 2 Visited num of GT in top 10 : 3.4295 Candidate List Size: 68.5878 / 300000 Target recall: 3.5 Search Time: 0.029482 Cen Search Time: 0.0175233 Table time: 0.0087846 Vec Search Time: 0.0031741 with repeat times: 0 / 3',
'Numcluster: 10 Cluster Batch: 1 Visited num of GT in top 10 : 3.5489 Candidate List Size: 76.1865 / 300000 Target recall: 3.5 Search Time: 0.029892 Cen Search Time: 0.0180813 Table time: 0.0086363 Vec Search Time: 0.0031744 with repeat times: 3 / 3',
'Numcluster: 9 Cluster Batch: 1 Visited num of GT in top 10 : 3.4295 Candidate List Size: 68.5878 / 300000 Target recall: 3.5 Search Time: 0.028916 Cen Search Time: 0.0172371 Table time: 0.0086213 Vec Search Time: 0.0030576 with repeat times: 0 / 3',
]

Data4 = ['Numcluster: 200 Cluster Batch: 20 Visited num of GT in top 10 : 4.7463 Candidate List Size: 2007.74 / 400000 Target recall: 3.5 Search Time: 0.172668 Cen Search Time: 0.137587 Table time: 0.0086332 Vec Search Time: 0.0264483 with repeat times: 0 / 3',
'Numcluster: 180 Cluster Batch: 20 Visited num of GT in top 10 : 4.7413 Candidate List Size: 1807.56 / 400000 Target recall: 3.5 Search Time: 0.16016 Cen Search Time: 0.127465 Table time: 0.0085787 Vec Search Time: 0.0241165 with repeat times: 0 / 3',
'Numcluster: 160 Cluster Batch: 20 Visited num of GT in top 10 : 4.7337 Candidate List Size: 1607.18 / 400000 Target recall: 3.5 Search Time: 0.147285 Cen Search Time: 0.116841 Table time: 0.008571 Vec Search Time: 0.0218738 with repeat times: 0 / 3',
'Numcluster: 140 Cluster Batch: 20 Visited num of GT in top 10 : 4.7251 Candidate List Size: 1406.9 / 400000 Target recall: 3.5 Search Time: 0.133903 Cen Search Time: 0.105765 Table time: 0.0086423 Vec Search Time: 0.0194952 with repeat times: 0 / 3',
'Numcluster: 120 Cluster Batch: 20 Visited num of GT in top 10 : 4.7119 Candidate List Size: 1206.62 / 400000 Target recall: 3.5 Search Time: 0.120599 Cen Search Time: 0.0948751 Table time: 0.0086194 Vec Search Time: 0.0171044 with repeat times: 0 / 3',
'Numcluster: 100 Cluster Batch: 20 Visited num of GT in top 10 : 4.6914 Candidate List Size: 1006.39 / 400000 Target recall: 3.5 Search Time: 0.10667 Cen Search Time: 0.0831327 Table time: 0.0086373 Vec Search Time: 0.0149004 with repeat times: 0 / 3',
'Numcluster: 80 Cluster Batch: 20 Visited num of GT in top 10 : 4.6598 Candidate List Size: 805.839 / 400000 Target recall: 3.5 Search Time: 0.0923388 Cen Search Time: 0.0712634 Table time: 0.0085796 Vec Search Time: 0.0124958 with repeat times: 0 / 3',
'Numcluster: 60 Cluster Batch: 20 Visited num of GT in top 10 : 4.6095 Candidate List Size: 605.168 / 400000 Target recall: 3.5 Search Time: 0.077371 Cen Search Time: 0.0586659 Table time: 0.0086281 Vec Search Time: 0.010077 with repeat times: 0 / 3',
'Numcluster: 40 Cluster Batch: 20 Visited num of GT in top 10 : 4.495 Candidate List Size: 403.913 / 400000 Target recall: 3.5 Search Time: 0.0610964 Cen Search Time: 0.0448309 Table time: 0.0085995 Vec Search Time: 0.007666 with repeat times: 0 / 3',
'Numcluster: 20 Cluster Batch: 20 Visited num of GT in top 10 : 4.1589 Candidate List Size: 202.557 / 400000 Target recall: 3.5 Search Time: 0.0423539 Cen Search Time: 0.0287584 Table time: 0.0085937 Vec Search Time: 0.0050018 with repeat times: 0 / 3',
'Numcluster: 1 Cluster Batch: 20 Visited num of GT in top 10 : 0.7929 Candidate List Size: 10.0489 / 400000 Target recall: 3.5 Search Time: 0.0155834 Cen Search Time: 0.0058872 Table time: 0.0086483 Vec Search Time: 0.0010479 with repeat times: 0 / 3',
'Numcluster: 19 Cluster Batch: 2 Visited num of GT in top 10 : 4.1282 Candidate List Size: 192.516 / 400000 Target recall: 3.5 Search Time: 0.0409013 Cen Search Time: 0.0274597 Table time: 0.0085777 Vec Search Time: 0.0048639 with repeat times: 3 / 3',
'Numcluster: 17 Cluster Batch: 2 Visited num of GT in top 10 : 4.0517 Candidate List Size: 172.298 / 400000 Target recall: 3.5 Search Time: 0.0387856 Cen Search Time: 0.0255706 Table time: 0.0086205 Vec Search Time: 0.0045945 with repeat times: 0 / 3',
'Numcluster: 15 Cluster Batch: 2 Visited num of GT in top 10 : 3.9526 Candidate List Size: 152.049 / 400000 Target recall: 3.5 Search Time: 0.0366261 Cen Search Time: 0.0236787 Table time: 0.0086161 Vec Search Time: 0.0043313 with repeat times: 0 / 3',
'Numcluster: 13 Cluster Batch: 2 Visited num of GT in top 10 : 3.8315 Candidate List Size: 131.892 / 400000 Target recall: 3.5 Search Time: 0.0344028 Cen Search Time: 0.0217422 Table time: 0.0086031 Vec Search Time: 0.0040575 with repeat times: 0 / 3',
'Numcluster: 11 Cluster Batch: 2 Visited num of GT in top 10 : 3.6811 Candidate List Size: 111.612 / 400000 Target recall: 3.5 Search Time: 0.032035 Cen Search Time: 0.0196696 Table time: 0.0086032 Vec Search Time: 0.0037622 with repeat times: 0 / 3',
'Numcluster: 9 Cluster Batch: 2 Visited num of GT in top 10 : 3.4872 Candidate List Size: 91.3763 / 400000 Target recall: 3.5 Search Time: 0.0292709 Cen Search Time: 0.0171969 Table time: 0.0086239 Vec Search Time: 0.0034501 with repeat times: 0 / 3',
'Numcluster: 10 Cluster Batch: 1 Visited num of GT in top 10 : 3.5913 Candidate List Size: 101.482 / 400000 Target recall: 3.5 Search Time: 0.0303843 Cen Search Time: 0.018215 Table time: 0.0085572 Vec Search Time: 0.0036121 with repeat times: 3 / 3',
'Numcluster: 9 Cluster Batch: 1 Visited num of GT in top 10 : 3.4872 Candidate List Size: 91.3763 / 400000 Target recall: 3.5 Search Time: 0.0293195 Cen Search Time: 0.0172916 Table time: 0.0085705 Vec Search Time: 0.0034574 with repeat times: 0 / 3',
]

Data5 = ['Numcluster: 200 Cluster Batch: 20 Visited num of GT in top 10 : 4.6858 Candidate List Size: 2507.64 / 500000 Target recall: 3.5 Search Time: 0.177749 Cen Search Time: 0.137507 Table time: 0.0086105 Vec Search Time: 0.0316313 with repeat times: 0 / 3',
'Numcluster: 180 Cluster Batch: 20 Visited num of GT in top 10 : 4.6804 Candidate List Size: 2257.8 / 500000 Target recall: 3.5 Search Time: 0.164726 Cen Search Time: 0.127332 Table time: 0.0085777 Vec Search Time: 0.0288163 with repeat times: 0 / 3',
'Numcluster: 160 Cluster Batch: 20 Visited num of GT in top 10 : 4.6733 Candidate List Size: 2007.65 / 500000 Target recall: 3.5 Search Time: 0.15146 Cen Search Time: 0.116835 Table time: 0.0085987 Vec Search Time: 0.0260264 with repeat times: 0 / 3',
'Numcluster: 140 Cluster Batch: 20 Visited num of GT in top 10 : 4.6642 Candidate List Size: 1757.6 / 500000 Target recall: 3.5 Search Time: 0.137576 Cen Search Time: 0.105916 Table time: 0.0085549 Vec Search Time: 0.023105 with repeat times: 0 / 3',
'Numcluster: 120 Cluster Batch: 20 Visited num of GT in top 10 : 4.6511 Candidate List Size: 1507.45 / 500000 Target recall: 3.5 Search Time: 0.123864 Cen Search Time: 0.094934 Table time: 0.0086045 Vec Search Time: 0.0203257 with repeat times: 0 / 3',
'Numcluster: 100 Cluster Batch: 20 Visited num of GT in top 10 : 4.6352 Candidate List Size: 1257.28 / 500000 Target recall: 3.5 Search Time: 0.108933 Cen Search Time: 0.08288 Table time: 0.0085519 Vec Search Time: 0.0175007 with repeat times: 0 / 3',
'Numcluster: 80 Cluster Batch: 20 Visited num of GT in top 10 : 4.609 Candidate List Size: 1006.86 / 500000 Target recall: 3.5 Search Time: 0.0942105 Cen Search Time: 0.0708981 Table time: 0.008635 Vec Search Time: 0.0146774 with repeat times: 0 / 3',
'Numcluster: 60 Cluster Batch: 20 Visited num of GT in top 10 : 4.559 Candidate List Size: 756.145 / 500000 Target recall: 3.5 Search Time: 0.0788179 Cen Search Time: 0.0585382 Table time: 0.0085507 Vec Search Time: 0.011729 with repeat times: 0 / 3',
'Numcluster: 40 Cluster Batch: 20 Visited num of GT in top 10 : 4.4516 Candidate List Size: 504.77 / 500000 Target recall: 3.5 Search Time: 0.0622483 Cen Search Time: 0.0447894 Table time: 0.0085872 Vec Search Time: 0.0088717 with repeat times: 0 / 3',
'Numcluster: 20 Cluster Batch: 20 Visited num of GT in top 10 : 4.1385 Candidate List Size: 253.067 / 500000 Target recall: 3.5 Search Time: 0.0429797 Cen Search Time: 0.0286425 Table time: 0.008652 Vec Search Time: 0.0056852 with repeat times: 0 / 3',
'Numcluster: 1 Cluster Batch: 20 Visited num of GT in top 10 : 0.8505 Candidate List Size: 12.5461 / 500000 Target recall: 3.5 Search Time: 0.0157633 Cen Search Time: 0.0058831 Table time: 0.0086133 Vec Search Time: 0.0012669 with repeat times: 0 / 3',
'Numcluster: 19 Cluster Batch: 2 Visited num of GT in top 10 : 4.1079 Candidate List Size: 240.52 / 500000 Target recall: 3.5 Search Time: 0.0415818 Cen Search Time: 0.0274167 Table time: 0.0085875 Vec Search Time: 0.0055776 with repeat times: 3 / 3',
'Numcluster: 17 Cluster Batch: 2 Visited num of GT in top 10 : 4.0425 Candidate List Size: 215.304 / 500000 Target recall: 3.5 Search Time: 0.0399505 Cen Search Time: 0.0260497 Table time: 0.0085691 Vec Search Time: 0.0053317 with repeat times: 0 / 3',
'Numcluster: 15 Cluster Batch: 2 Visited num of GT in top 10 : 3.9538 Candidate List Size: 190 / 500000 Target recall: 3.5 Search Time: 0.0377354 Cen Search Time: 0.0241964 Table time: 0.0085822 Vec Search Time: 0.0049568 with repeat times: 0 / 3',
'Numcluster: 13 Cluster Batch: 2 Visited num of GT in top 10 : 3.839 Candidate List Size: 164.749 / 500000 Target recall: 3.5 Search Time: 0.0348929 Cen Search Time: 0.0217125 Table time: 0.0086139 Vec Search Time: 0.0045665 with repeat times: 0 / 3',
'Numcluster: 11 Cluster Batch: 2 Visited num of GT in top 10 : 3.7021 Candidate List Size: 139.419 / 500000 Target recall: 3.5 Search Time: 0.0324842 Cen Search Time: 0.0196244 Table time: 0.0086372 Vec Search Time: 0.0042226 with repeat times: 0 / 3',
'Numcluster: 9 Cluster Batch: 2 Visited num of GT in top 10 : 3.5128 Candidate List Size: 114.082 / 500000 Target recall: 3.5 Search Time: 0.0297586 Cen Search Time: 0.0172703 Table time: 0.0085998 Vec Search Time: 0.0038885 with repeat times: 0 / 3',
'Numcluster: 7 Cluster Batch: 2 Visited num of GT in top 10 : 3.2468 Candidate List Size: 88.738 / 500000 Target recall: 3.5 Search Time: 0.0269154 Cen Search Time: 0.0148471 Table time: 0.0086031 Vec Search Time: 0.0034652 with repeat times: 0 / 3',
'Numcluster: 8 Cluster Batch: 1 Visited num of GT in top 10 : 3.3918 Candidate List Size: 101.422 / 500000 Target recall: 3.5 Search Time: 0.0283795 Cen Search Time: 0.0160303 Table time: 0.0086414 Vec Search Time: 0.0037078 with repeat times: 3 / 3',
]


Data6 = ['Numcluster: 200 Cluster Batch: 20 Visited num of GT in top 10 : 4.6299 Candidate List Size: 3009.91 / 600000 Target recall: 3.5 Search Time: 0.180944 Cen Search Time: 0.135629 Table time: 0.0085889 Vec Search Time: 0.036726 with repeat times: 0 / 3',
'Numcluster: 180 Cluster Batch: 20 Visited num of GT in top 10 : 4.626 Candidate List Size: 2709.95 / 600000 Target recall: 3.5 Search Time: 0.167391 Cen Search Time: 0.125573 Table time: 0.0085667 Vec Search Time: 0.0332519 with repeat times: 0 / 3',
'Numcluster: 160 Cluster Batch: 20 Visited num of GT in top 10 : 4.6188 Candidate List Size: 2409.75 / 600000 Target recall: 3.5 Search Time: 0.153735 Cen Search Time: 0.115125 Table time: 0.0085725 Vec Search Time: 0.0300371 with repeat times: 0 / 3',
'Numcluster: 140 Cluster Batch: 20 Visited num of GT in top 10 : 4.6101 Candidate List Size: 2109.43 / 600000 Target recall: 3.5 Search Time: 0.139295 Cen Search Time: 0.104099 Table time: 0.0085547 Vec Search Time: 0.0266413 with repeat times: 0 / 3',
'Numcluster: 120 Cluster Batch: 20 Visited num of GT in top 10 : 4.5979 Candidate List Size: 1809.09 / 600000 Target recall: 3.5 Search Time: 0.125176 Cen Search Time: 0.0932914 Table time: 0.0085615 Vec Search Time: 0.0233228 with repeat times: 0 / 3',
'Numcluster: 100 Cluster Batch: 20 Visited num of GT in top 10 : 4.5827 Candidate List Size: 1508.82 / 600000 Target recall: 3.5 Search Time: 0.110324 Cen Search Time: 0.0817369 Table time: 0.0085672 Vec Search Time: 0.0200203 with repeat times: 0 / 3',
'Numcluster: 80 Cluster Batch: 20 Visited num of GT in top 10 : 4.5552 Candidate List Size: 1208.21 / 600000 Target recall: 3.5 Search Time: 0.0952163 Cen Search Time: 0.0700026 Table time: 0.0085656 Vec Search Time: 0.0166481 with repeat times: 0 / 3',
'Numcluster: 60 Cluster Batch: 20 Visited num of GT in top 10 : 4.5152 Candidate List Size: 907.307 / 600000 Target recall: 3.5 Search Time: 0.0794901 Cen Search Time: 0.057713 Table time: 0.0085775 Vec Search Time: 0.0131996 with repeat times: 0 / 3',
'Numcluster: 40 Cluster Batch: 20 Visited num of GT in top 10 : 4.4175 Candidate List Size: 605.731 / 600000 Target recall: 3.5 Search Time: 0.0623804 Cen Search Time: 0.0440728 Table time: 0.0085692 Vec Search Time: 0.0097384 with repeat times: 0 / 3',
'Numcluster: 20 Cluster Batch: 20 Visited num of GT in top 10 : 4.1314 Candidate List Size: 303.694 / 600000 Target recall: 3.5 Search Time: 0.0431306 Cen Search Time: 0.028283 Table time: 0.0085549 Vec Search Time: 0.0062927 with repeat times: 0 / 3',
'Numcluster: 1 Cluster Batch: 20 Visited num of GT in top 10 : 0.8968 Candidate List Size: 15.0238 / 600000 Target recall: 3.5 Search Time: 0.0157635 Cen Search Time: 0.0057767 Table time: 0.0085618 Vec Search Time: 0.001425 with repeat times: 0 / 3',
'Numcluster: 19 Cluster Batch: 2 Visited num of GT in top 10 : 4.1006 Candidate List Size: 288.635 / 600000 Target recall: 3.5 Search Time: 0.041746 Cen Search Time: 0.0270541 Table time: 0.0085756 Vec Search Time: 0.0061163 with repeat times: 3 / 3',
'Numcluster: 17 Cluster Batch: 2 Visited num of GT in top 10 : 4.0387 Candidate List Size: 258.345 / 600000 Target recall: 3.5 Search Time: 0.03947 Cen Search Time: 0.0251643 Table time: 0.0085539 Vec Search Time: 0.0057518 with repeat times: 0 / 3',
'Numcluster: 15 Cluster Batch: 2 Visited num of GT in top 10 : 3.955 Candidate List Size: 228.024 / 600000 Target recall: 3.5 Search Time: 0.0372708 Cen Search Time: 0.0233088 Table time: 0.0085359 Vec Search Time: 0.0054261 with repeat times: 0 / 3',
'Numcluster: 13 Cluster Batch: 2 Visited num of GT in top 10 : 3.8511 Candidate List Size: 197.726 / 600000 Target recall: 3.5 Search Time: 0.0351249 Cen Search Time: 0.0215521 Table time: 0.0085657 Vec Search Time: 0.0050071 with repeat times: 0 / 3',
'Numcluster: 11 Cluster Batch: 2 Visited num of GT in top 10 : 3.7224 Candidate List Size: 167.328 / 600000 Target recall: 3.5 Search Time: 0.0324731 Cen Search Time: 0.0192775 Table time: 0.0085814 Vec Search Time: 0.0046142 with repeat times: 0 / 3',
'Numcluster: 9 Cluster Batch: 2 Visited num of GT in top 10 : 3.5385 Candidate List Size: 136.893 / 600000 Target recall: 3.5 Search Time: 0.0296477 Cen Search Time: 0.0168786 Table time: 0.0085812 Vec Search Time: 0.0041879 with repeat times: 0 / 3',
'Numcluster: 7 Cluster Batch: 2 Visited num of GT in top 10 : 3.2773 Candidate List Size: 106.493 / 600000 Target recall: 3.5 Search Time: 0.0268613 Cen Search Time: 0.0145442 Table time: 0.0085684 Vec Search Time: 0.0037487 with repeat times: 0 / 3',
'Numcluster: 8 Cluster Batch: 1 Visited num of GT in top 10 : 3.4206 Candidate List Size: 121.7 / 600000 Target recall: 3.5 Search Time: 0.0282134 Cen Search Time: 0.0156896 Table time: 0.0085597 Vec Search Time: 0.0039641 with repeat times: 3 / 3',
]


Data7 = ['Numcluster: 200 Cluster Batch: 20 Visited num of GT in top 10 : 4.572 Candidate List Size: 3511.23 / 700000 Target recall: 3.5 Search Time: 0.187874 Cen Search Time: 0.135682 Table time: 0.0085842 Vec Search Time: 0.043608 with repeat times: 0 / 3',
'Numcluster: 180 Cluster Batch: 20 Visited num of GT in top 10 : 4.5685 Candidate List Size: 3161.38 / 700000 Target recall: 3.5 Search Time: 0.17323 Cen Search Time: 0.125099 Table time: 0.0086376 Vec Search Time: 0.0394928 with repeat times: 0 / 3',
'Numcluster: 160 Cluster Batch: 20 Visited num of GT in top 10 : 4.5633 Candidate List Size: 2811.23 / 700000 Target recall: 3.5 Search Time: 0.15857 Cen Search Time: 0.114575 Table time: 0.0086117 Vec Search Time: 0.0353827 with repeat times: 0 / 3',
'Numcluster: 140 Cluster Batch: 20 Visited num of GT in top 10 : 4.5555 Candidate List Size: 2460.97 / 700000 Target recall: 3.5 Search Time: 0.144055 Cen Search Time: 0.103753 Table time: 0.0086539 Vec Search Time: 0.0316479 with repeat times: 0 / 3',
'Numcluster: 120 Cluster Batch: 20 Visited num of GT in top 10 : 4.5469 Candidate List Size: 2110.53 / 700000 Target recall: 3.5 Search Time: 0.129013 Cen Search Time: 0.0927213 Table time: 0.0086029 Vec Search Time: 0.0276885 with repeat times: 0 / 3',
'Numcluster: 100 Cluster Batch: 20 Visited num of GT in top 10 : 4.5333 Candidate List Size: 1760.26 / 700000 Target recall: 3.5 Search Time: 0.113628 Cen Search Time: 0.0813624 Table time: 0.0086459 Vec Search Time: 0.0236202 with repeat times: 0 / 3',
'Numcluster: 80 Cluster Batch: 20 Visited num of GT in top 10 : 4.5066 Candidate List Size: 1409.47 / 700000 Target recall: 3.5 Search Time: 0.0977471 Cen Search Time: 0.0696413 Table time: 0.0086266 Vec Search Time: 0.0194792 with repeat times: 0 / 3',
'Numcluster: 60 Cluster Batch: 20 Visited num of GT in top 10 : 4.4708 Candidate List Size: 1058.4 / 700000 Target recall: 3.5 Search Time: 0.081451 Cen Search Time: 0.0573984 Table time: 0.0086423 Vec Search Time: 0.0154103 with repeat times: 0 / 3',
'Numcluster: 40 Cluster Batch: 20 Visited num of GT in top 10 : 4.3789 Candidate List Size: 706.709 / 700000 Target recall: 3.5 Search Time: 0.0637508 Cen Search Time: 0.0438304 Table time: 0.0085912 Vec Search Time: 0.0113292 with repeat times: 0 / 3',
'Numcluster: 20 Cluster Batch: 20 Visited num of GT in top 10 : 4.1186 Candidate List Size: 354.238 / 700000 Target recall: 3.5 Search Time: 0.043779 Cen Search Time: 0.0281341 Table time: 0.008573 Vec Search Time: 0.0070719 with repeat times: 0 / 3',
'Numcluster: 1 Cluster Batch: 20 Visited num of GT in top 10 : 0.9301 Candidate List Size: 17.5419 / 700000 Target recall: 3.5 Search Time: 0.0160653 Cen Search Time: 0.0057993 Table time: 0.0086418 Vec Search Time: 0.0016242 with repeat times: 0 / 3',
'Numcluster: 19 Cluster Batch: 2 Visited num of GT in top 10 : 4.0914 Candidate List Size: 336.657 / 700000 Target recall: 3.5 Search Time: 0.0423595 Cen Search Time: 0.0268677 Table time: 0.0086176 Vec Search Time: 0.0068742 with repeat times: 3 / 3',
'Numcluster: 17 Cluster Batch: 2 Visited num of GT in top 10 : 4.0271 Candidate List Size: 301.329 / 700000 Target recall: 3.5 Search Time: 0.040123 Cen Search Time: 0.0250991 Table time: 0.0085923 Vec Search Time: 0.0064316 with repeat times: 0 / 3',
'Numcluster: 15 Cluster Batch: 2 Visited num of GT in top 10 : 3.9522 Candidate List Size: 265.953 / 700000 Target recall: 3.5 Search Time: 0.0377984 Cen Search Time: 0.0231707 Table time: 0.0086288 Vec Search Time: 0.0059989 with repeat times: 0 / 3',
'Numcluster: 13 Cluster Batch: 2 Visited num of GT in top 10 : 3.8534 Candidate List Size: 230.602 / 700000 Target recall: 3.5 Search Time: 0.035461 Cen Search Time: 0.0212741 Table time: 0.0086022 Vec Search Time: 0.0055847 with repeat times: 0 / 3',
'Numcluster: 11 Cluster Batch: 2 Visited num of GT in top 10 : 3.7287 Candidate List Size: 195.188 / 700000 Target recall: 3.5 Search Time: 0.0329779 Cen Search Time: 0.0192789 Table time: 0.0085833 Vec Search Time: 0.0051157 with repeat times: 0 / 3',
'Numcluster: 9 Cluster Batch: 2 Visited num of GT in top 10 : 3.5547 Candidate List Size: 159.691 / 700000 Target recall: 3.5 Search Time: 0.0299686 Cen Search Time: 0.0168051 Table time: 0.0085581 Vec Search Time: 0.0046054 with repeat times: 0 / 3',
'Numcluster: 7 Cluster Batch: 2 Visited num of GT in top 10 : 3.3074 Candidate List Size: 124.21 / 700000 Target recall: 3.5 Search Time: 0.0272257 Cen Search Time: 0.0145124 Table time: 0.0086303 Vec Search Time: 0.004083 with repeat times: 0 / 3',
'Numcluster: 8 Cluster Batch: 1 Visited num of GT in top 10 : 3.443 Candidate List Size: 141.965 / 700000 Target recall: 3.5 Search Time: 0.0287422 Cen Search Time: 0.0155425 Table time: 0.0088204 Vec Search Time: 0.0043793 with repeat times: 3 / 3',
]

Data8 = ['Numcluster: 200 Cluster Batch: 20 Visited num of GT in top 10 : 4.5247 Candidate List Size: 4013.64 / 800000 Target recall: 3.5 Search Time: 0.192694 Cen Search Time: 0.135271 Table time: 0.0085794 Vec Search Time: 0.0488433 with repeat times: 0 / 3',
'Numcluster: 180 Cluster Batch: 20 Visited num of GT in top 10 : 4.521 Candidate List Size: 3613.76 / 800000 Target recall: 3.5 Search Time: 0.178398 Cen Search Time: 0.125331 Table time: 0.0086309 Vec Search Time: 0.0444362 with repeat times: 0 / 3',
'Numcluster: 160 Cluster Batch: 20 Visited num of GT in top 10 : 4.516 Candidate List Size: 3213.56 / 800000 Target recall: 3.5 Search Time: 0.163836 Cen Search Time: 0.115512 Table time: 0.0085972 Vec Search Time: 0.0397271 with repeat times: 0 / 3',
'Numcluster: 140 Cluster Batch: 20 Visited num of GT in top 10 : 4.5092 Candidate List Size: 2813.16 / 800000 Target recall: 3.5 Search Time: 0.148218 Cen Search Time: 0.104295 Table time: 0.0086011 Vec Search Time: 0.0353224 with repeat times: 0 / 3',
'Numcluster: 120 Cluster Batch: 20 Visited num of GT in top 10 : 4.4998 Candidate List Size: 2412.51 / 800000 Target recall: 3.5 Search Time: 0.132575 Cen Search Time: 0.0933422 Table time: 0.0086483 Vec Search Time: 0.0305846 with repeat times: 0 / 3',
'Numcluster: 100 Cluster Batch: 20 Visited num of GT in top 10 : 4.4872 Candidate List Size: 2012.03 / 800000 Target recall: 3.5 Search Time: 0.116258 Cen Search Time: 0.0815784 Table time: 0.0085825 Vec Search Time: 0.0260966 with repeat times: 0 / 3',
'Numcluster: 80 Cluster Batch: 20 Visited num of GT in top 10 : 4.4636 Candidate List Size: 1610.96 / 800000 Target recall: 3.5 Search Time: 0.100336 Cen Search Time: 0.070082 Table time: 0.0085303 Vec Search Time: 0.0217235 with repeat times: 0 / 3',
'Numcluster: 60 Cluster Batch: 20 Visited num of GT in top 10 : 4.4308 Candidate List Size: 1209.68 / 800000 Target recall: 3.5 Search Time: 0.0835233 Cen Search Time: 0.0576369 Table time: 0.0086421 Vec Search Time: 0.0172443 with repeat times: 0 / 3',
'Numcluster: 40 Cluster Batch: 20 Visited num of GT in top 10 : 4.3451 Candidate List Size: 807.741 / 800000 Target recall: 3.5 Search Time: 0.0660598 Cen Search Time: 0.0447687 Table time: 0.0086167 Vec Search Time: 0.0126744 with repeat times: 0 / 3',
'Numcluster: 20 Cluster Batch: 20 Visited num of GT in top 10 : 4.0946 Candidate List Size: 404.956 / 800000 Target recall: 3.5 Search Time: 0.0446117 Cen Search Time: 0.0280393 Table time: 0.0086657 Vec Search Time: 0.0079067 with repeat times: 0 / 3',
'Numcluster: 1 Cluster Batch: 20 Visited num of GT in top 10 : 0.9645 Candidate List Size: 20.0571 / 800000 Target recall: 3.5 Search Time: 0.0161893 Cen Search Time: 0.0058042 Table time: 0.0086089 Vec Search Time: 0.0017762 with repeat times: 0 / 3',
'Numcluster: 19 Cluster Batch: 2 Visited num of GT in top 10 : 4.0658 Candidate List Size: 384.855 / 800000 Target recall: 3.5 Search Time: 0.0431438 Cen Search Time: 0.0270357 Table time: 0.0086178 Vec Search Time: 0.0074903 with repeat times: 3 / 3',
'Numcluster: 17 Cluster Batch: 2 Visited num of GT in top 10 : 4.0071 Candidate List Size: 344.467 / 800000 Target recall: 3.5 Search Time: 0.0406721 Cen Search Time: 0.0250995 Table time: 0.0086006 Vec Search Time: 0.006972 with repeat times: 0 / 3',
'Numcluster: 15 Cluster Batch: 2 Visited num of GT in top 10 : 3.9342 Candidate List Size: 304.035 / 800000 Target recall: 3.5 Search Time: 0.0383827 Cen Search Time: 0.0232982 Table time: 0.0085852 Vec Search Time: 0.0064993 with repeat times: 0 / 3',
'Numcluster: 13 Cluster Batch: 2 Visited num of GT in top 10 : 3.8383 Candidate List Size: 263.627 / 800000 Target recall: 3.5 Search Time: 0.0358793 Cen Search Time: 0.0212443 Table time: 0.0086382 Vec Search Time: 0.0059968 with repeat times: 0 / 3',
'Numcluster: 11 Cluster Batch: 2 Visited num of GT in top 10 : 3.7217 Candidate List Size: 223.116 / 800000 Target recall: 3.5 Search Time: 0.0334063 Cen Search Time: 0.0192394 Table time: 0.0086641 Vec Search Time: 0.0055028 with repeat times: 0 / 3',
'Numcluster: 9 Cluster Batch: 2 Visited num of GT in top 10 : 3.5487 Candidate List Size: 182.555 / 800000 Target recall: 3.5 Search Time: 0.0304893 Cen Search Time: 0.0168933 Table time: 0.0086176 Vec Search Time: 0.0049784 with repeat times: 0 / 3',
'Numcluster: 7 Cluster Batch: 2 Visited num of GT in top 10 : 3.3124 Candidate List Size: 141.993 / 800000 Target recall: 3.5 Search Time: 0.0275505 Cen Search Time: 0.0145372 Table time: 0.0086105 Vec Search Time: 0.0044028 with repeat times: 0 / 3',
'Numcluster: 8 Cluster Batch: 1 Visited num of GT in top 10 : 3.44 Candidate List Size: 162.304 / 800000 Target recall: 3.5 Search Time: 0.02897 Cen Search Time: 0.0157174 Table time: 0.0085777 Vec Search Time: 0.0046749 with repeat times: 3 / 3',
]


Data10 = ['Numcluster: 200 Cluster Batch: 20 Visited num of GT in top 10 : 4.4763 Candidate List Size: 5018.02 / 1000000 Target recall: 3.5 Search Time: 0.213295 Cen Search Time: 0.137252 Table time: 0.0085685 Vec Search Time: 0.0674748 with repeat times: 0 / 3',
'Numcluster: 180 Cluster Batch: 20 Visited num of GT in top 10 : 4.4725 Candidate List Size: 4518.04 / 1000000 Target recall: 3.5 Search Time: 0.19669 Cen Search Time: 0.127436 Table time: 0.0085821 Vec Search Time: 0.0606726 with repeat times: 0 / 3',
'Numcluster: 160 Cluster Batch: 20 Visited num of GT in top 10 : 4.4685 Candidate List Size: 4017.65 / 1000000 Target recall: 3.5 Search Time: 0.180722 Cen Search Time: 0.11657 Table time: 0.0086009 Vec Search Time: 0.0555513 with repeat times: 0 / 3',
'Numcluster: 140 Cluster Batch: 20 Visited num of GT in top 10 : 4.4626 Candidate List Size: 3516.96 / 1000000 Target recall: 3.5 Search Time: 0.162176 Cen Search Time: 0.105677 Table time: 0.0085387 Vec Search Time: 0.0479604 with repeat times: 0 / 3',
'Numcluster: 120 Cluster Batch: 20 Visited num of GT in top 10 : 4.4545 Candidate List Size: 3016.16 / 1000000 Target recall: 3.5 Search Time: 0.145136 Cen Search Time: 0.0945885 Table time: 0.0085338 Vec Search Time: 0.0420139 with repeat times: 0 / 3',
'Numcluster: 100 Cluster Batch: 20 Visited num of GT in top 10 : 4.4414 Candidate List Size: 2515.47 / 1000000 Target recall: 3.5 Search Time: 0.126242 Cen Search Time: 0.0825527 Table time: 0.0086071 Vec Search Time: 0.0350818 with repeat times: 0 / 3',
'Numcluster: 80 Cluster Batch: 20 Visited num of GT in top 10 : 4.4228 Candidate List Size: 2014.13 / 1000000 Target recall: 3.5 Search Time: 0.108889 Cen Search Time: 0.0710911 Table time: 0.0085682 Vec Search Time: 0.0292296 with repeat times: 0 / 3',
'Numcluster: 60 Cluster Batch: 20 Visited num of GT in top 10 : 4.3889 Candidate List Size: 1512.4 / 1000000 Target recall: 3.5 Search Time: 0.0892437 Cen Search Time: 0.0585018 Table time: 0.0085562 Vec Search Time: 0.0221857 with repeat times: 0 / 3',
'Numcluster: 40 Cluster Batch: 20 Visited num of GT in top 10 : 4.3051 Candidate List Size: 1010 / 1000000 Target recall: 3.5 Search Time: 0.070147 Cen Search Time: 0.044688 Table time: 0.0085304 Vec Search Time: 0.0169286 with repeat times: 0 / 3',
'Numcluster: 20 Cluster Batch: 20 Visited num of GT in top 10 : 4.0636 Candidate List Size: 506.342 / 1000000 Target recall: 3.5 Search Time: 0.0471096 Cen Search Time: 0.0284596 Table time: 0.0085088 Vec Search Time: 0.0101412 with repeat times: 0 / 3',
'Numcluster: 1 Cluster Batch: 20 Visited num of GT in top 10 : 1.0192 Candidate List Size: 25.1737 / 1000000 Target recall: 3.5 Search Time: 0.0165257 Cen Search Time: 0.0058744 Table time: 0.0085711 Vec Search Time: 0.0020802 with repeat times: 0 / 3',
'Numcluster: 19 Cluster Batch: 2 Visited num of GT in top 10 : 4.0385 Candidate List Size: 481.178 / 1000000 Target recall: 3.5 Search Time: 0.0455284 Cen Search Time: 0.0269316 Table time: 0.0086118 Vec Search Time: 0.009985 with repeat times: 3 / 3',
'Numcluster: 17 Cluster Batch: 2 Visited num of GT in top 10 : 3.9816 Candidate List Size: 430.675 / 1000000 Target recall: 3.5 Search Time: 0.0424923 Cen Search Time: 0.025366 Table time: 0.0085214 Vec Search Time: 0.0086049 with repeat times: 0 / 3',
'Numcluster: 15 Cluster Batch: 2 Visited num of GT in top 10 : 3.9129 Candidate List Size: 380.142 / 1000000 Target recall: 3.5 Search Time: 0.0404594 Cen Search Time: 0.023934 Table time: 0.0085862 Vec Search Time: 0.0079392 with repeat times: 0 / 3',
'Numcluster: 13 Cluster Batch: 2 Visited num of GT in top 10 : 3.8258 Candidate List Size: 329.607 / 1000000 Target recall: 3.5 Search Time: 0.0378128 Cen Search Time: 0.0216878 Table time: 0.0086013 Vec Search Time: 0.0075237 with repeat times: 0 / 3',
'Numcluster: 11 Cluster Batch: 2 Visited num of GT in top 10 : 3.7211 Candidate List Size: 279.031 / 1000000 Target recall: 3.5 Search Time: 0.0346 Cen Search Time: 0.0194082 Table time: 0.0086099 Vec Search Time: 0.0065819 with repeat times: 0 / 3',
'Numcluster: 9 Cluster Batch: 2 Visited num of GT in top 10 : 3.5535 Candidate List Size: 228.324 / 1000000 Target recall: 3.5 Search Time: 0.0319715 Cen Search Time: 0.0174882 Table time: 0.0085931 Vec Search Time: 0.0058902 with repeat times: 0 / 3',
'Numcluster: 7 Cluster Batch: 2 Visited num of GT in top 10 : 3.3254 Candidate List Size: 177.589 / 1000000 Target recall: 3.5 Search Time: 0.0286819 Cen Search Time: 0.014946 Table time: 0.0085897 Vec Search Time: 0.0051462 with repeat times: 0 / 3',
'Numcluster: 8 Cluster Batch: 1 Visited num of GT in top 10 : 3.449 Candidate List Size: 202.994 / 1000000 Target recall: 3.5 Search Time: 0.0297683 Cen Search Time: 0.0156851 Table time: 0.0085692 Vec Search Time: 0.005514 with repeat times: 3 / 3',
]

Recall_index = 13
CL_index = 19
plt.figure()
ClusterNum = []
Recall = []

for i in range(len(Data10)):
    List = Data10[i].split(" ")
    ClusterNum.append(float(List[1]))
    Recall.append(float(List[Recall_index]))

plt.scatter(ClusterNum, Recall, marker = "o")

plt.xlabel("ClusterNum")
plt.ylabel("Recall10@10")
plt.title("SIFT1B with 2 Million NC")
plt.show()




Result = []
length = 11

for i in range(length):
    List = Data1[i].split(" ")
    Result.append([[float(List[CL_index])], [float(List[Recall_index])]])

for i in range(length):
    List = Data2[i].split(" ")
    Result[i][0].append(float(List[CL_index]))
    Result[i][1].append(float(List[Recall_index]))

for i in range(len(Result)):
    List = Data3[i].split(" ")
    Result[i][0].append(float(List[CL_index]))
    Result[i][1].append(float(List[Recall_index]))

for i in range(len(Result)):
    List = Data4[i].split(" ")
    Result[i][0].append(float(List[CL_index]))
    Result[i][1].append(float(List[Recall_index]))

for i in range(len(Result)):
    List = Data5[i].split(" ")
    Result[i][0].append(float(List[CL_index]))
    Result[i][1].append(float(List[Recall_index]))

for i in range(len(Result)):
    List = Data6[i].split(" ")
    Result[i][0].append(float(List[CL_index]))
    Result[i][1].append(float(List[Recall_index]))

for i in range(len(Result)):
    List = Data7[i].split(" ")
    Result[i][0].append(float(List[CL_index]))
    Result[i][1].append(float(List[Recall_index]))

for i in range(len(Result)):
    List = Data8[i].split(" ")
    Result[i][0].append(float(List[CL_index]))
    Result[i][1].append(float(List[Recall_index]))

for i in range(len(Result)):
    List = Data10[i].split(" ")
    Result[i][0].append(float(List[CL_index]))
    Result[i][1].append(float(List[Recall_index]))

for i in range(len(Result)):
    plt.plot(np.array(Result[i][0]) * 1000, Result[i][1], marker = "o")

    plt.xlabel("Dataset Scale")
    plt.ylabel("Acheived Recall")
    plt.title("SIFT1B with 2 Million NC and m = 150")
    plt.show()

