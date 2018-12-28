import os
os.chdir(r'C:\Users\31236\Desktop\baseline\LAL')

from Classes.results import Results

res2plot = Results()
res2plot.readResult('DatasetBreast-exp')
res2plot.plotResults(metrics = ['accuracy'])