import matplotlib.pyplot as plt
import numpy as np

x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
y = np.array([80.68, 80.53, 80.45, 80.33, 80.27, 80.20, 80.13, 80.06, 79.92])

plt.scatter(x, y, marker='o', color='red')
plt.xlabel('weight_lambda')
plt.ylabel('NLI Validation Accuracy')
plt.title('NLI Validation Accuracy vs weight_lambda')

plt.grid(True)

plt.savefig('nli_val_vs_weight_lambda.png')
plt.show()