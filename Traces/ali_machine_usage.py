#!/usr/bin/python3

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import os


#Collect data
x_values = []
y_values = []
with open("ali_machine_usage.dat", 'r') as f:
    data = f.readlines()
    y_values = list(map(float, data[0].split(",")))
    x_values = np.arange(len(data[0].split(",")))

#BarWidth = 0.55

Figure = plt.figure(figsize=(3,2))
Graph = Figure.add_subplot(111)
PDF = PdfPages("ali_machine_usage.pdf")

plt.plot(x_values, y_values, '-c', label="avg")
#plt.title("Percentage Disk I/O per Machine", fontsize=14)

YLabel = plt.ylabel("Storage BW Util (%)", multialignment='center', fontsize=12)
YLabel.set_position((0.0,0.5))
YLabel.set_linespacing(0.5)

#Want the labels to be days, and the increments are on a 10s basis (hence 360 not 3600)
Xticks = np.arange(0,len(x_values),360*24)
Graph.set_xticks(Xticks)
Graph.set_xticklabels(['0','1','2','3','4','5','6','7'],fontsize=11)
Graph.xaxis.set_ticks_position('none')
Graph.set_xlabel('Time (days)', fontsize=14)

YTicks = np.arange(0,40,5)
Graph.set_yticks(YTicks)
#What are the action labels?
Graph.set_yticklabels(['0', '5', '10', '15', '20', '25', '30', '35'],fontsize=11)
#Where are the ticks?
Graph.yaxis.set_ticks_position('none')

Graph.set_axisbelow(True)
Graph.yaxis.grid(color='lightgrey', linestyle='solid')

#lg = Graph.legend(loc='upper right', prop={'size':9}, ncol=1, borderaxespad=0.2)
#lg.draw_frame(False)

Graph.grid(b=True, which='minor')

Graph.set_xlim(0, len(x_values))
Graph.set_ylim((0,35))

PDF.savefig(Figure, bbox_inches='tight')
#plt.savefig("ali_machine_usage.png")

PDF.close()
