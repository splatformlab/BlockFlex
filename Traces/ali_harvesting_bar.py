#!/usr/bin/python3

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
from operator import truediv

def label_rects(rects):
    """
    Attach a text label above each bar displaying the height
    """
    for rect in rects:
        height = round(rect.get_height(),2)
        Graph.text(rect.get_x() + rect.get_width()/2.0, 1.01 * height, f"{height}", ha='center', va='bottom', fontsize=6,weight='bold')


hr_1 = []
hr_6 = []
hr_12 = []
hr_72 = []

with open('ali_harvesting_bar.dat', 'r') as f:
    data = f.readlines()
    for i in range(3):
        hr_1.append(np.mean(list(filter(lambda x: x!= 0,map(float, data[i*4+0].split(","))))))
        hr_6.append(np.mean(list(filter(lambda x: x!= 0,map(float, data[i*4+1].split(","))))))
        hr_12.append(np.mean(list(filter(lambda x: x!= 0,map(float, data[i*4+2].split(","))))))
        hr_72.append(np.mean(list(filter(lambda x: x!= 0,map(float, data[i*4+3].split(","))))))


x_labels = ['10%','25%','50%']
y_labels = ['0', '', '20','', '40', '', '60', '', '80', '', '100']
x_values = np.arange(len(x_labels))

matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams.update({'font.family': 'serif'})
matplotlib.rcParams['xtick.major.pad'] = '8'
matplotlib.rcParams['ytick.major.pad'] = '8'
matplotlib.rcParams['hatch.linewidth'] = 0.5

#TODO CHECK THIS Fig Size
Figure = plt.figure( figsize=(3.0,1) )
Graph = Figure.add_subplot(111)
PDF = PdfPages( "ali_harvesting_bar.pdf" )

#Graph.set_title("Survival Rate of Different Harvest Sizes")

BarWidth = 0.15

#Don't need to read, have the data here locally
#DataFile = open( "ali_surv.dat", "r" )
#Data = DataFile.readlines()

#YValues = map( float, Data[ 0 ].split() )

YLabel = plt.ylabel( '% of VMs', fontsize=7 )
#YLabel.set_position( (0.0, 1.0) )
#YLabel.set_linespacing( 0.5 )


#XTicks = XValues + BarWidth / 2
Graph.set_xticks( x_values )
Graph.set_xticklabels( x_labels, fontsize=7, ha="center")
Graph.xaxis.set_ticks_position( 'none' )
Graph.tick_params(pad=0)
#YLabel.set_linespacing( 0.5 )
#Graph.set_xlabel( 'Comparison Schemes')



YTicks = np.arange(0,1.1,0.1)
Graph.set_yticks( YTicks )
Graph.set_yticklabels(y_labels, fontsize=7 )
Graph.yaxis.set_ticks_position( 'none' )

Graph.set_axisbelow(True)
#TODO do I want this?
Graph.yaxis.grid(color = 'lightgray', linestyle= 'solid')

color1, color2, color3, color4 = '#EC7497', '#fac205', '#95d0fc', '#96f97b'

Rects1 = Graph.bar(x_values - BarWidth, hr_1, BarWidth, edgecolor='black', label="1 hr",hatch="--", color=color1)
Rects2 = Graph.bar(x_values, hr_6, BarWidth, edgecolor='black', label="6 hrs", hatch="", color=color2)
Rects3 = Graph.bar(x_values + BarWidth, hr_12, BarWidth, edgecolor='black', label="12 hrs", hatch="////", color=color3)
Rects3 = Graph.bar(x_values + 2*BarWidth, hr_72, BarWidth, edgecolor='black', label="3 days", hatch="\\"*4, color='lightcyan')

#label_rects(Rects1)
#label_rects(Rects2)
#label_rects(Rects3)

#YValues = map( float, Data[ 0 ].split() )
#Graph.bar( XValues, YValues, BarWidth, edgecolor='black', color=color1, hatch="", label="")


Graph.set_xlim( ( BarWidth-1, len(x_values)) )
Graph.set_ylim( ( 0, 1.05 ) )
Graph.set_xlabel('Storage capacity', fontsize=7)


#Graph.text(2, -75000, '(c) Audio', fontsize=14)


lg=Graph.legend(prop={'size':5}, ncol=4, loc='upper center', borderaxespad=0.)
lg.draw_frame(False)

PDF.savefig( Figure, bbox_inches='tight' )
#plt.savefig("ali_surv.png")
PDF.close()


