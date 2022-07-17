import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

names = ['216652-000', '216652-001']
ml_preds = [0.524, 0.641]
time_ml = [0.602, 0.7449]
time_nonml = [60.366]
# log axis?

#plt.plot(time_ml, names, 'o', alpha = 0.5, color='b') # rotate names and put on x?

# paired bar charts
print('average predictions', np.average(ml_preds))
print('average ml', np.average(time_ml))
print('average no ml', np.average(time_nonml))

x = np.arange(len(names))
width = 0.2
  
# plot data in grouped manner of bar type
plt.bar(x-0.2, ml_preds, width, color='cyan')
plt.bar(x, time_ml, width, color='orange')
plt.bar(x+0.2, time_nonml, width, color='green')
plt.xticks(x, names)
plt.title("Source Selection Time Comparison")
plt.xlabel("image name (may 26, 2020)")
plt.ylabel("time (seconds)")
plt.legend(["ml time (just prediction)", "ml time", "non-ml time"])
plt.show()