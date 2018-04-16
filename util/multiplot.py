#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import argparse, sys

parser = argparse.ArgumentParser()
parser.add_argument('-x', '--xcolumn', help="Column to use as X-axis", type=int, required=True)
parser.add_argument('-c', '--column', help="Column to use as Y-axis", type=int, required=True)
parser.add_argument('files', nargs='+')
parser.add_argument('--xfilt', help="Ignore any rows with X value larger", type=float)
parser.add_argument('--xoffset', help="Add this to all X values", type=float)
parser.add_argument('-o', '--outfile', help="output file to save plot as")
parser.add_argument('--label', help="Label for legend", action='append')
parser.add_argument('--dpi', help="Resolution to save output as", default=300)
parser.add_argument('-t', '--title')
parser.add_argument('--xlab', help="Label for X-axis")
parser.add_argument('--ylab', help="Label for Y-axis")
parser.add_argument('-m', '--max', help="Set maximum Y value", type=float)
parser.add_argument('--xmax', help="Set maximum X value", type=float)
parser.add_argument('--xmin', help="Set minimum X value", type=float)
parser.add_argument('-s', '--show', help="Display plot to screen", action="store_true")
parser.add_argument('-l', '--log', help="Take log of y value", action="store_true")
parser.add_argument('-p', '--poly', help="Subtract polynomial of order P from data", type=int)

#parser.add_argument('-v', dest='verbose', action='store_true')

args = parser.parse_args()

show=(args.show or (args.outfile==None))

title = args.title
ymax = args.max

subPoly = args.poly

data = []
for file in args.files:
    d=np.genfromtxt(file, dtype=float)
    data.append(d)

def subtractFit(y, x, order=1):
    fit = np.polyfit(x,y,order)
    poly = np.poly1d(fit)
    fitY = poly(x)
    return(y-fitY)
    
def doplot(d, xcol, ycol, label=None):
    xcol -= 1
    ycol -= 1
    x = d[:,xcol]
    y = d[:,ycol]

    if (args.xoffset!=None):
        x += args.xoffset

    if (args.xfilt!=None):
        index = x<=args.xfilt
        x = x[index]
        y = y[index]
    
    if args.log:
        y = np.log(y)

    if subPoly != None:
        y = subtractFit(y, x, subPoly)
        
    plt.plot(x, y, label=label)

if args.label==None:
    label=[None]*len(data)
else:
    label = args.label
    
for d,f in zip(data,label):
    doplot(d, args.xcolumn, args.column, f)

    
if args.label!=None: plt.legend()

if (title!=None):
    plt.title(title)
if (args.xlab!=None):
    plt.xlabel(args.xlab)
if (args.ylab!=None):
    plt.ylabel(args.ylab)

if args.xmin!=None or args.xmax!=None:
    (xmin, xmax) = plt.xlim()
    if args.xmin!=None:
         xmin = args.xmin
    if args.xmax!=None:
        xmax = args.xmax
    plt.xlim(xmin, xmax)

#plt.tight_layout()

if (args.outfile!=None):
    plt.savefig(args.outfile, dpi=args.dpi, transparent=True)

if show:
    plt.show()
