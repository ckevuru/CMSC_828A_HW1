import matplotlib.pyplot as plt
import numpy as np

x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
y = np.array([97.42, 97.67, 97.75, 97.84, 97.94, 98.12, 98.21, 98.29, 98.36])

plt.scatter(x, y, marker='o')
plt.xlabel('weight_lambda')
plt.ylabel('NER Validation Accuracy')
plt.title('NER Validation Accuracy vs weight_lambda')

plt.grid(True)

plt.savefig('ner_val_vs_weight_lambda.png')
plt.show()