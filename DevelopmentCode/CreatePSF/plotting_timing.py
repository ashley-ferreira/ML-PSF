import matplotlib.pyplot as plt
import numpy as np

names = ['0216652-000', '0216652-000', '0216652-000', '0216652-000', '0216652-000', '0216652-000', '0216652-000']
time_ml = []
time_nonml = []

#plt.plot(time_ml, names, 'o', alpha = 0.5, color='b') # rotate names and put on x?

# paired bar charts

print('average ml', np.average(time_ml))
print('average no ml', np.average(time_nonml))