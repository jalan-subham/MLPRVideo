import os 
import glob 
import gc
# import torch
# torch.cuda.empty_cache()
# gc.collect()
# user_count = len(glob.glob("BagOfLies/Finalised/User_*"))
# # # 
# for i in range(user_count):
#     # os.system(f"python3 au_presence_descriptive_pt_loo.py {i}")
#     os.system(f"python3 of_landmarks_descriptive.py {i}")
'''
0.6666666666666666
0.5
0.4444444444444444
0.5
0.6
0.7
0.8
0.7
0.6
0.7
0.7
0.8
1.0
0.5
0.9
0.5
0.7
0.7
0.7
0.7
0.8
0.6
0.9
0.625
0.6
0.5555555555555556
0.75
0.8571428571428571
0.5
0.6
0.8
0.5454545454545454
0.4
0.6
0.625'''
accs = []
with open("OFEyeDescriptive2D-SVM.txt", "r") as file:
    lines = file.readlines()
    print(lines)
    for line in lines:
        accs.append(float(line.strip()))
print(sum(accs)/len(accs))

# with open("AUPresenceLSTM_mean.txt") as file:
#     line = [float(x.strip()) for x in file.readlines()]
#     print(sum(line)/len(line))