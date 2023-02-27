#!/bin/bash

#SBATCH -n 40 # 指定核心数量
#SBATCH -N 1 # 指定node的数量
#SBATCH -t 0-1:00 # 运行总时间，天数-小时数-分钟， D-HH:MM
#SBATCH --mem=8000 # 所有核心可以使用的内存池大小，MB为单位
#SBATCH -o myjob.o # 把输出结果STDOUT保存在哪一个文件
#SBATCH -e myjob.e # 把报错结果STDERR保存在哪一个文件


./AblationPS