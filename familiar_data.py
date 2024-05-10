import glob 
import cv2
import tqdm 
import pandas as pd 
import numpy as np 
paths = glob.glob("BagOfLies/Finalised/User_*/run_*/openface/*.csv")
# annotations = pd.read_csv("BagOfLies/Annotations.csv")
# vid_annotations = annotations["video"].str.replace("./Finalised", "BagOfLies/Finalised").str.replace("/video.mp4", "")
# print(vid_annotations)
# paths = glob.glob("BagOfLies/Finalised/User_*/run_*/video.mp4")
# y = []

# import matplotlib.pyplot as plt

# video_lengths = []
# for path in tqdm.tqdm(paths):
#     cap = cv2.VideoCapture(path)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     length = frame_count / fps
#     video_lengths.append(length)
#     video_run = "/".join(path.split("/")[:-1])
#     print(video_run)
#     ind = vid_annotations[vid_annotations == video_run].index
#     curr_y = annotations.iloc[ind]["truth"].values[0]
#     y.append(curr_y)
#     cap.release()

#     # Plot histogram for y = 0
# plt.hist([length for length, label in zip(video_lengths, y) if label == 0], bins=10, color='blue', alpha=0.5, label='y = 0')

# # Plot histogram for y = 1
# plt.hist([length for length, label in zip(video_lengths, y) if label == 1], bins=10, color='red', alpha=0.5, label='y = 1')

# plt.xlabel('Video Length')
# plt.ylabel('Frequency')
# plt.title('Histogram of Video Lengths for y = 0 and y = 1')
# plt.legend()
# plt.savefig("plt_hist.png")

# path = paths[0]
# df = np.load('/'.join(path.split('/')[:-2]) + "/landmarks.npy")
# print(df.shape)
# df = pd.DataFrame(df)
# print(df)
# df.to_csv("eye_gaze_stats.csv")

pca_2d_landmarks = pd.read_csv("2d_landmarks_pca_setb.csv").drop(["truth", "paths"],axis=1)
print(pca_2d_landmarks)