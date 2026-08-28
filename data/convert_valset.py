# -*- coding: utf-8 -*-
# 在服务器上执行：转换官方验证集 → jalmbench jsonl
import json, os
from collections import Counter
d = json.load(open("/root/lalm_framing_revision_v6/data/text_behaviors_val_set.json"))
os.makedirs("/root/lalm_framing_revision_v6/data/jalmbench", exist_ok=True)
rows = []
for k in d:
    for item in d[k]:
        test_case = item.get("test_case") or item.get("behavior")
        gen = item.get("generation") or item.get("response")
        if not test_case or not gen:
            continue
        humans = [item.get("human_%d" % i) for i in range(3)]
        humans = [h for h in humans if h is not None]
        if not humans:
            continue
        label = int(Counter(humans).most_common(1)[0][0])
        rows.append({"behavior": test_case, "response": gen, "label": label,
                     "source": "harmbench_official",
                     "behavior_id": item.get("behavior_id")})
with open("/root/lalm_framing_revision_v6/data/jalmbench/text_behaviors_val_set.jsonl",
          "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("转换完成:", len(rows), "条")
print("label 分布:", dict(Counter(r["label"] for r in rows)))
