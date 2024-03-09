import matplotlib.pyplot as plt
import numpy as np

x = np.array([1, 2, 4, 6, 8, 10])
y = np.array([98.12, 98.10, 98.11, 98.09, 98.13, 98.12])

plt.scatter(x, y, marker='o')
plt.xlabel('sigma')
plt.ylabel('NER Validation Accuracy')
plt.title('NER Validation Accuracy vs sigma')

plt.grid(True)
plt.ylim([97, 99])
plt.savefig('ner_val_vs_sigma.png')
plt.show()