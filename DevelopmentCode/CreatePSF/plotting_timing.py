import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 0.9 conf cutoff

names = ['216652-000', '216652-001', '216652-002', '216652-003', '216652-004', '216652-005', '216652-006']
ml_preds = [0.5049, 0.627114, 0.6088, 0.6094, 0.6114, 0.6723, 0.627]
time_ml = [0.5796, 0.72742, 0.71030, 0.7161, 0.703265, 0.777109, 0.827]
time_nonml = [61.096, 118.0005, 133.888, 158.418, 85.749, 177.521, 148.296]
# log axis?

#plt.plot(time_ml, names, 'o', alpha = 0.5, color='b') # rotate names and put on x?

# paired bar charts
print('average predictions', np.average(ml_preds))
print('average ml', np.average(time_ml))
print('average no ml', np.average(time_nonml))
print('average ml / no ml', np.average(time_ml)/np.average(time_nonml))

x = np.arange(len(names))
width = 0.2
  
# plot data in grouped manner of bar type
plt.bar(x-0.2, ml_preds, width, alpha=0.5)
plt.bar(x, time_ml, width, alpha=0.7)
plt.bar(x+0.2, time_nonml, width, color='purple', alpha=0.5)
plt.yscale('log')
plt.ylim(0,10**3)
plt.xticks(x, names)
plt.title("Source Selection Time Comparison")
plt.xlabel("image name (may 26, 2020)")
plt.ylabel("time (seconds)")
plt.legend(["CNN model prediction only time", "CNN model process time", "non-CNN proccess time"], loc='upper left')
plt.show()