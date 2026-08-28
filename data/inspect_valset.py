import json
d = json.load(open("text_behaviors_val_set.json"))
print("type:", type(d))
if isinstance(d, dict):
    print("keys:", list(d.keys())[:10])
    for k in list(d.keys())[:3]:
        v = d[k]
        print("key=", k, "type=", type(v), "len=", len(v) if hasattr(v, "__len__") else "?")
        if isinstance(v, list) and len(v) > 0:
            print("  sample:", str(v[0])[:400])
elif isinstance(d, list):
    print("len:", len(d))
    print("sample:", str(d[0])[:400])
