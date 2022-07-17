import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

names = ['216652-000', '216652-001']
ml_preds = [0.5049, 0.627114]
time_ml = [0.5796, 0.72742]
time_nonml = [61.096, 118.0005]
# log axis?

#plt.plot(time_ml, names, 'o', alpha = 0.5, color='b') # rotate names and put on x?

# paired bar charts
print('average predictions', np.average(ml_preds))
print('average ml', np.average(time_ml))
print('average no ml', np.average(time_nonml))

x = np.arange(len(names))
width = 0.2
  
# plot data in grouped manner of bar type
plt.bar(x-0.2, ml_preds, width, alpha=0.5)
plt.bar(x, time_ml, width, alpha=0.7)
plt.bar(x+0.2, time_nonml, width, color='purple', alpha=0.5)
plt.yscale('log')
plt.xticks(x, names)
plt.title("Source Selection Time Comparison")
plt.xlabel("image name (may 26, 2020)")
plt.ylabel("time (seconds)")
plt.legend(["ml prediction only time", "ml process time", "non-ml proccess time"])
plt.show()