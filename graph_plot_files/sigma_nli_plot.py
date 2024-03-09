import matplotlib.pyplot as plt
import numpy as np

x = np.array([1, 2, 4, 6, 8, 10])
y = np.array([80.33, 80.30, 80.29, 80.28, 80.30, 80.29])

plt.scatter(x, y, marker='o', color='red')
plt.xlabel('sigma')
plt.ylabel('NLI Validation Accuracy')
plt.title('NLI Validation Accuracy vs sigma')

plt.grid(True)
plt.ylim([79, 81])
plt.savefig('nli_val_vs_sigma.png')
plt.show()