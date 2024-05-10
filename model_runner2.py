import os 
import glob 
import gc
import tqdm
# import torch
# torch.cuda.empty_cache()
# gc.collect()
user_count = len(glob.glob("BagOfLies/Finalised/User_*"))
# # 
vids = glob.glob("BagOfLies/Finalised/User_*/run_*/video.mp4")
accs_ = []
for j in range(1):
    print(f"Running {j}")
    for i in tqdm.tqdm(range(user_count)):
        # os.system(f"python3 au_presence_descriptive_pt_loo.py {i}")
        os.system(f"python3 of_landmarks_descriptive.py {i}")
    with open("OFLandmarksDescriptiveGaze-SVM.txt", "r") as file:
        accs = [float(x.strip()) for x in file.readlines()
                ]
        accs_.append(sum(accs)/len(accs))
    print(accs_)
    os.remove("OFLandmarksDescriptiveGaze-SVM.txt")

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