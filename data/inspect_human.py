import json
d = json.load(open("text_behaviors_val_set.json"))
# human_0/1/2 取值分布 + cls 分布
from collections import Counter
h0, h1, h2, cls = Counter(), Counter(), Counter(), Counter()
both = Counter()  # human 三票 vs cls
n = 0
for k in d:
    for item in d[k]:
        n += 1
        h0[str(item.get("human_0"))] += 1
        h1[str(item.get("human_1"))] += 1
        h2[str(item.get("human_2"))] += 1
        cls[str(item.get("cls"))] += 1
print("n =", n)
print("human_0:", dict(h0))
print("human_1:", dict(h1))
print("human_2:", dict(h2))
print("cls:", dict(cls))
# 少数样例
for k in list(d.keys())[:2]:
    for item in d[k][:1]:
        print("sample human:", item.get("human_0"), item.get("human_1"), item.get("human_2"), "cls:", item.get("cls"))
