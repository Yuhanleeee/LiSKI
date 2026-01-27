import json
import math

def l2_normalize_dict(d):
    norm = math.sqrt(sum(v * v for v in d.values()))
    if norm == 0:
        return d  
    return {k: v / norm for k, v in d.items()}


file_path = 'grad_score_qwen253b.json'
with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

data = [l2_normalize_dict(d) for d in data]

sum_dict = data[0]
for dict_temp in data[1:]:
    for layer_id in sum_dict:
        sum_dict[layer_id] += dict_temp[layer_id]

sorted_items = sorted(sum_dict.items(), key=lambda x: x[1], reverse=True)
top24_layers = [item[0] for item in sorted_items[:24]]
print(top24_layers)
