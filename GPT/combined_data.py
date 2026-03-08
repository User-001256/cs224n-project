'''
Run this to create a combined dataset with the real shakespeare sonnets and the 200
generated ones.
'''
import re

def read_sonnets(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    sonnets = re.split(r"\n\s*\d+\s*\n", text)[1:]
    return [s.strip() for s in sonnets if s.strip()]

def write_sonnets(file_path, sonnets):
    with open(file_path, "w", encoding="utf-8") as f:
        for i, sonnet in enumerate(sonnets, start=1):
            f.write(f"{i}\n\n{sonnet}\n\n")

real_sonnets = read_sonnets("../data/sonnets.txt")
synthetic_sonnets = read_sonnets("../data/sonnets_clean.txt")

combined_sonnets = real_sonnets * 2 + synthetic_sonnets

write_sonnets("../data/combined_sonnets_train.txt", combined_sonnets)
