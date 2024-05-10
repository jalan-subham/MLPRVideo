import os 
import glob 
import gc
# import torch
# torch.cuda.empty_cache()
# gc.collect()
user_count = len(glob.glob("BagOfLies/Finalised/User_*"))
# 
for i in range(user_count):
    os.system(f"python3 of_landmarks_descriptive.py {i}")

accs = []
with open("OFLandmarksDescriptive-GNB.txt", "r") as file:
    lines = file.readlines()
    for line in lines:
        accs.append(float(line.strip()))
print(sum(accs)/len(accs))