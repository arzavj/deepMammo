import numpy as np
import matplotlib
import matplotlib.pyplot as plt

deep_acc = [0.912088, 0.917582, 0.879121, 0.824176]
deep_p = [0.903226, 0.904255, 0.870968, 0.953846]
deep_r = [0.923077, 0.934066, 0.89011, 0.681319]

shallow_acc = [0.82967, 0.851648, 0.912088, 0.906593]
shallow_p = [0.905405, 0.863636, 0.941176, 0.9204]
shallow_r = [0.736264, 0.835165, 0.879121, 0.8901]

x_axis = [0.25, 0.5, 0.75, 1.0]

list.reverse(deep_acc)
list.reverse(deep_p)
list.reverse(deep_r)

fig = plt.figure()                                                               
ax = fig.add_subplot(1,1,1)
ax.set_xticks(np.arange(0.25, 1.1, 0.25))                                                       
ax.grid(which='both') 

ax.plot(x_axis, deep_acc)
ax.plot(x_axis, deep_p)
ax.plot(x_axis, deep_r)

ax.plot(x_axis, shallow_acc, '--')
ax.plot(x_axis, shallow_p, '--')
ax.plot(x_axis, shallow_r, '--')

plt.xlabel('Fraction of full training process')
plt.legend(['Deeper Accuracy', 'Deeper Precision', 'Deeper Recall', 'Shallow Accuracy', 'Shallow Precision', 'Shallow Recall'], loc='lower right')
plt.savefig('deepershallow.png')