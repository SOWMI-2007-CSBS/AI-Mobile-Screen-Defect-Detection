import os
from collections import Counter

folders = [
    "dataset/train/labels",
    "dataset/valid/labels",
    "dataset/test/labels"
]

counts = Counter()

for folder in folders:
    if not os.path.exists(folder):
        print("Folder not found:", folder)
        continue

    for file in os.listdir(folder):
        if file.endswith(".txt"):
            path = os.path.join(folder, file)

            with open(path, "r") as f:
                for line in f:
                    parts = line.strip().split()

                    if parts:
                        class_id = int(parts[0])
                        counts[class_id] += 1

print("\nClass ID Count:")
print("----------------")

for class_id in sorted(counts):
    print("Class", class_id, ":", counts[class_id])

print("\nDone!")