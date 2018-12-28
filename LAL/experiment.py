import os
os.chdir(r'D:\baseline\LAL')

from Classes.results import Results

filename = './exp/checkerboard2x2-exp.p'
res2plot = Results()
res2plot.readResult('checkerboard2x2-exp')
res2plot.plotResults(metrics = ['accuracy'])