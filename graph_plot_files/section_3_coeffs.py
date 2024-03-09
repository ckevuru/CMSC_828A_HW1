import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau

task_1 = {
    'Math': {'Electrical': 22.92, 'Global Facts': 29.29, 'Biology': 21.04, 'Geography': 23.86, 'Econometrics': 27.43, 'Formal Logic': 33.60},
    'Electrical': {'Math': 21.19, 'Global Facts': 30.30, 'Biology': 25.89, 'Geography': 32.49, 'Econometrics': 27.43, 'Formal Logic': 31.20},
    'Global Facts': {'Electrical': 25.69, 'Math': 26.02, 'Biology': 17.15, 'Geography': 18.78, 'Econometrics': 25.60, 'Formal Logic': 25.60},
    'Biology': {'Electrical': 24.69, 'Math': 27.88, 'Global Facts': 26.26, 'Geography': 34.01, 'Econometrics': 27.43, 'Formal Logic': 34.40},
    'Geography': {'Electrical': 24.31, 'Math': 25.28, 'Global Facts': 24.24, 'Biology': 30.74, 'Econometrics': 26.55, 'Formal Logic': 33.60},
    'Econometrics': {'Electrical': 22.92, 'Math': 23.42, 'Global Facts': 33.33, 'Biology': 18.45, 'Geography': 24.37, 'Formal Logic': 29.60},
    'Formal Logic': {'Electrical': 24.31, 'Math': 23.79, 'Global Facts': 31.31, 'Biology': 16.50, 'Geography': 19.29, 'Econometrics': 25.66}
}

task_2 = {
    'Math': {'Electrical': 23.61, 'Global Facts': 17.17, 'Biology': 32.04, 'Geography': 21.32, 'Econometrics': 26.55, 'Formal Logic': 18.40},
    'Electrical': {'Math': 26.02, 'Global Facts': 25.25, 'Biology': 31.72, 'Geography': 26.90, 'Econometrics': 25.66, 'Formal Logic': 14.40},
    'Global Facts': {'Electrical': 20.83, 'Math': 24.91, 'Biology': 21.36, 'Geography': 21.32, 'Econometrics': 23.12, 'Formal Logic': 19.20},
    'Biology': {'Electrical': 25.00, 'Math': 27.88, 'Global Facts': 20.20, 'Geography': 25.38, 'Econometrics': 24.78, 'Formal Logic': 16.80},
    'Geography': {'Electrical': 22.22, 'Math': 23.79, 'Global Facts': 14.14, 'Biology': 31.72, 'Econometrics': 26.55, 'Formal Logic': 17.60},
    'Econometrics': {'Electrical': 23.61, 'Math': 27.14, 'Global Facts': 23.23, 'Biology': 31.72, 'Geography': 25.89, 'Formal Logic': 15.20},
    'Formal Logic': {'Electrical': 24.31, 'Math': 27.14, 'Global Facts': 19.19, 'Biology': 30.42, 'Geography': 23.86, 'Econometrics': 23.89}
}

icl = {
    'Math': {'Electrical': 20.19, 'Global Facts': 22.25, 'Biology': 19.25, 'Geography': 20.28, 'Econometrics': 24.90, 'Formal Logic': 25.50},
    'Electrical': {'Math': 23.24, 'Global Facts': 28.76, 'Biology': 24.22, 'Geography': 31.09, 'Econometrics': 26.89, 'Formal Logic': 28.92},
    'Global Facts': {'Electrical': 19.73, 'Math': 22.54, 'Biology': 16.56, 'Geography': 23.42, 'Econometrics': 24.86, 'Formal Logic': 27.12},
    'Biology': {'Electrical': 22.16, 'Math': 25.74, 'Global Facts': 27.32, 'Geography': 23.56, 'Econometrics': 26.54, 'Formal Logic': 28.41},
    'Geography': {'Electrical': 21.45, 'Math': 24.67, 'Global Facts': 20.68, 'Biology': 29.87, 'Econometrics': 27.12, 'Formal Logic': 29.10},
    'Econometrics': {'Electrical': 23.21, 'Math': 25.67, 'Global Facts': 33.57, 'Biology': 24.88, 'Geography': 22.14, 'Formal Logic': 26.90},
    'Formal Logic': {'Electrical': 20.87, 'Math': 25.78, 'Global Facts': 32.80, 'Biology': 24.45, 'Geography': 16.55, 'Econometrics': 26.22}
}

zero_shot = {
    'Math': 21.56,
    'Electrical': 24.31,
    'Global Facts': 23.23,
    'Biology': 26.57,
    'Geography': 18.27,
    'Econometrics': 23.01,
    'Formal Logic': 26.40
}

def get_accuracy_gains(task):
    result = []
    for subject, values in task.items():
        for inner_subject, score in values.items():
            result.append(score - zero_shot[inner_subject])
    return result

accuracy_gain_task1_zshot = get_accuracy_gains(task_1)
accuracy_gain_task2_zshot = get_accuracy_gains(task_2)
accuracy_gain_icl_zshot = get_accuracy_gains(icl)

def get_same_sign_freq(arr1, arr2):
    res = 0
    for a1, a2 in zip(arr1, arr2):
        if a1*a2>=0:
            res+=1
    return res

print(f"Task1-ICL same sign freq: {get_same_sign_freq(accuracy_gain_task1_zshot, accuracy_gain_icl_zshot)}")
print()
print(f"Task2-ICL same sign freq: {get_same_sign_freq(accuracy_gain_task2_zshot, accuracy_gain_icl_zshot)}")
print()

def get_coefficients(accuracy_gain_task1, accuracy_gain_task2):
    # Calculate Pearson correlation coefficient
    pearson_corr, _ = pearsonr(accuracy_gain_task1, accuracy_gain_task2)

    # Calculate Spearman correlation coefficient
    spearman_corr, _ = spearmanr(accuracy_gain_task1, accuracy_gain_task2)

    # Calculate Kendall's tau
    kendall_tau, _ = kendalltau(accuracy_gain_task1, accuracy_gain_task2)

    print("Pearson correlation coefficient:", pearson_corr)
    print("Spearman correlation coefficient:", spearman_corr)
    print("Kendall's tau:", kendall_tau)

get_coefficients(accuracy_gain_task1_zshot, accuracy_gain_icl_zshot)
print()
get_coefficients(accuracy_gain_task2_zshot, accuracy_gain_icl_zshot)