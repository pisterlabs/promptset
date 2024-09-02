import re

with open("log.txt", "r", encoding="utf-8") as f:
    log = f.read()


text = log
print(len(text))
# print(text[:1000])

# TOTAL: ~0.8 for single prompt optimization
# + ~ 0.04 for seed prompt generation and synthetic data generation
matches = re.findall(r"Model: (\w.+-\d+b)[\s\S]*?Total Cost Estimate: ([\d.eE+-]+)", text)

model_costs = {}
for match in matches:
    model = match[0]
    cost = float(match[1])
    model_costs[model] = model_costs.get(match[0], 0) + cost
    
print(model_costs)