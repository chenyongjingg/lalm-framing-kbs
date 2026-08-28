import json
d = json.load(open("text_behaviors_val_set.json"))
# 遍历所有 behavior 收集带 generation 的条目，看完整字段
keys = list(d.keys())
total_entries = 0
labels = {}
for k in keys:
    for item in d[k]:
        total_entries += 1
        lab = item.get("label")
        if lab is not None:
            labels[str(lab)] = labels.get(str(lab), 0) + 1
print("total entries:", total_entries)
print("label 分布:", labels)
# 打印一条完整记录的所有字段
for k in keys:
    for item in d[k]:
        if "generation" in item:
            print("字段:", list(item.keys()))
            print("样本 label=", item.get("label"))
            print("test_case[:200]:", str(item.get("test_case"))[:200])
            print("generation[:200]:", str(item.get("generation"))[:200])
            break
    break
# 多少条有 generation
withgen = sum(1 for k in keys for item in d[k] if item.get("generation"))
print("有 generation 的条目:", withgen, "/", total_entries)
