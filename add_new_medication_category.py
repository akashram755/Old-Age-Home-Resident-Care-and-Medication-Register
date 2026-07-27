import json

NEW_MEDICATION = "Clopidogrel 75mg"

with open("data.json") as f:
    records = json.load(f)

candidates = [r for r in records if r["resident_id"] != ""]
candidates.sort(key=lambda r: (r["date"], r["resident_id"]))


step = max(1, len(candidates) // 8)
chosen_ids = {candidates[i]["record_id"] for i in range(0, len(candidates), step)}

changed = 0
for r in records:
    if r["record_id"] in chosen_ids:
        r["medication"] = NEW_MEDICATION
        changed += 1

with open("data.json", "w") as f:
    json.dump(records, f, indent=2)

print(f"Set medication = '{NEW_MEDICATION}' on {changed} records, spread across "
      f"multiple residents and dates.")
print("Now run: python3 train_model.py")
