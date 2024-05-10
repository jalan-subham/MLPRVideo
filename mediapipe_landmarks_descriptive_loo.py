import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import glob
import pandas as pd
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from utils2 import *
import warnings
import numpy as np
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import sys 
from sklearn.metrics import accuracy_score

import scipy 

warnings.filterwarnings("ignore")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Using", device)


class MediapipeLandmarks(Dataset):

    def __init__(self, paths, transform=None):
        self.paths = []
        for path in paths:
            self.paths += glob.glob(path + "run_*/landmarks.npy")
        self.annotations = pd.read_csv("BagOfLies/Annotations.csv")
        self.vid_annotations = self.annotations["video"].str.replace("./Finalised", "BagOfLies/Finalised").str.replace("/video.mp4", "")
        self.transform = transform
        # self.max_seq_len = 1265
        # cols = pd.read_csv(self.paths[0]).columns
        # self.start_col = np.where(cols == " AU01_c")[0][0]
        # self.end_col = np.where(cols == " AU45_c")[0][0]

    def load_expressions(self, file_path):
        df = np.load(file_path)
        if not len(df):
            return None
        df = np.where(df > 0.1, 1, 0)
        mean = np.mean(df, axis=0)
        std = np.std(df, axis=0)
        maximum = np.max(df, axis=0)
        minimum = np.min(df, axis=0)
        median = np.median(df, axis=0)
        var = np.var(df, axis=0)
        per_25 = np.percentile(df, 25, axis=0)
        per_50 = np.percentile(df, 50, axis=0)
        per_75 = np.percentile(df, 75, axis=0)
        skew = scipy.stats.skew(df, axis=0)
        kurtosis = scipy.stats.kurtosis(df, axis=0)

        # print(mean.shape)
        df = np.concatenate((mean, std, maximum, minimum, median, var, per_25, per_50, per_75, kurtosis, skew), axis=0)
        df = np.nan_to_num(df, 0)
        return torch.tensor(df)

    def get_ground_truth(self, file_path):
        video_run = "/".join(file_path.split("/")[:-1])
        ind = self.vid_annotations[self.vid_annotations == video_run].index
        return self.annotations.iloc[ind]["truth"].values[0]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        X = self.load_expressions(self.paths[index])
        if X is None:
            return None, None
        y = self.get_ground_truth(self.paths[index])
        if self.transform:
            return self.transform(X.to(torch.float32)), y
        else:
            return X.to(torch.float32), y  # return data, label (X, y)


# class AUPresenceNN(nn.Module):
#     def __init__(self, input_shape, hidden_shape, output_shape):
#         super().__init__()
#         # self.lstm = nn.LSTM(input_shape, hidden_shape, batch_first=True)
#         self.dense = nn.Sequential(
#             nn.Linear(input_shape, 128),
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(64, 32),
#             nn.ReLU(),
#             nn.Linear(32, output_shape)
#         )

#     def forward(self, x, lengths):
#         # packed_input = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
#         # packed_output, _ = self.lstm(packed_input)
#         # output, _ = pad_packed_sequence(packed_output, batch_first=True)
#         # last_output = output[:, -1, :]  # Get the last output from each sequence
#         # # out = self.dense(last_output)
#         # out = last_output
#         out = self.dense(x)
#         return out


def collate_pad(batch):
    """Pads sequences to the maximum length within the batch."""
    X_batch, y_batch = zip(*batch)
    lengths = [len(seq) for seq in X_batch]
    X_padded = pad_sequence(X_batch, batch_first=True, padding_value=0)
    return X_padded, lengths, torch.tensor(y_batch)


paths = glob.glob("BagOfLies/Finalised/User_*/")
paths.sort()
i = int(sys.argv[1])
train_paths = paths[:i] + paths[i+1:]
test_paths = [paths[i]]

train_dataset = MediapipeLandmarks(train_paths)
test_dataset = MediapipeLandmarks(test_paths)

# print(f"Training set size: {len(train_dataset)}")
# print(f"Validation set size: {len(test_dataset)}")
# train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_pad, num_workers=2)
# test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True, collate_fn=collate_pad, num_workers=2)

# X_batch, lengths, y_batch = next(iter(train_dataloader))
# print("Sample Instance:")
# print(X_batch)
# print(f"X_batch shape: {X_batch.shape}")
# print(f"y_batch shape: {y_batch.shape}")

# model = AUPresenceNN(198, 2, 2).to(device)

# NUM_EPOCHS = 10
# loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
# # model = nn.DataParallel(model) 

# start_time = timer()

# # Train model
# model_results = train(model=model,
#                       train_dataloader=train_dataloader,
#                       test_dataloader=test_dataloader,
#                       optimizer=optimizer,
#                       loss_fn=loss_fn,
#                       epochs=NUM_EPOCHS)

# # End the timer and print out how long it took
# end_time = timer()
# print(f"Total training time: {end_time-start_time:.3f} seconds")

# plot_loss_curves(model_results)
# plt.savefig('outputs/AUPresenceNN' + str(i))
# with open('AUPresenceNN_AUC_last.txt', 'a') as f:
#     f.write(str(np.array(model_results["test_auc"][-1])) + '\n')

from sklearn.svm import SVC

X_train = []
y_train = []
X_test = []
y_test = []
for idx in range(len(train_dataset)):
    X, y = train_dataset[idx]
    if X is not None:
        X_train.append(X.numpy())
        y_train.append(y)
X_train = np.array(X_train)
y_train = np.array(y_train)

for idx in range(len(test_dataset)):
    X, y = test_dataset[idx]
    if X is not None:
        X_test.append(X.numpy())
        y_test.append(y)
X_test = np.array(X_test)
if X_test.ndim == 1:
    print(X_test)
    print(X_test.shape)
    X_test = X_test.reshape(1, -1)
y_test = np.array(y_test)

svc = SVC()
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"Accuracy: {acc}")

with open("accuracies/LandmarksSVC.txt", "a") as file:
    file.write(str(acc) + '\n')
