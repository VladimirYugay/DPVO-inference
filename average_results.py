import json
from collections import defaultdict
import numpy as np


path = "tum_eval"
average_results = defaultdict(list)
for seed in [0, 1, 2]:
    with open(f"{path}_{seed}/average_results.json", "r") as file:
        results = json.load(file)
    for key, value in results.items():
        average_results[key].append(value)

for key in average_results:
    average_results[key] = np.mean(average_results[key])
    print(key, average_results[key])