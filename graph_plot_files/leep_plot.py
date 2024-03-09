import matplotlib.pyplot as plt
import numpy as np

leap_scores = np.array([-0.5925, -0.6274, -0.6456])
accuracies = np.array([86.19, 77.87, 77.30])

plt.scatter(leap_scores[0], accuracies[0], label='Pair 1', marker='o')
plt.scatter(leap_scores[1], accuracies[1], label='Pair 2', marker='s')
plt.scatter(leap_scores[2], accuracies[2], label='Pair 3', marker='^')
plt.xlabel('LEEP Score')
plt.ylabel('Validation Accuracy')
plt.legend(['zero-shot baseline', 'domain adapted', 'task adapted'])
plt.title('Validation Accuracy vs LEEP score')

plt.grid(True)

plt.savefig('accuracy_vs_leep_score.png')
plt.show()