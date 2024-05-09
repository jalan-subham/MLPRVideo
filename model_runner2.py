import os 
import glob 
import gc
import tqdm
# import torch
# torch.cuda.empty_cache()
# gc.collect()
user_count = len(glob.glob("BagOfLies/Finalised/User_*"))
# # 
for i in tqdm.tqdm(range(user_count)):
    # os.system(f"python3 au_presence_descriptive_pt_loo.py {i}")
    os.system(f"python3 ensemble_2dface_gaze.py {i}")

# accs = []
# with open("accuracies/LandmarksSVC.txt", "r") as file:
#     lines = file.readlines()
#     print(lines)
#     for line in lines:
#         accs.append(float(line.strip()))
# print(sum(accs)/len(accs))

# with open("AUPresenceLSTM_mean.txt") as file:
#     line = [float(x.strip()) for x in file.readlines()]
#     print(sum(line)/len(line))