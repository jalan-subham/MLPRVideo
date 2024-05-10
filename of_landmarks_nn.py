import torch
import pandas as pd
import numpy as np
from torch import nn
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import glob 
import sys
import scipy
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings("ignore")
from sklearn.svm import SVC
from utils_basic import *
from timeit import default_timer as timer
device = "cuda:1"
class NN(nn.Module):
    def __init__(self, input_shape, hidden_shape, output_shape):
        super().__init__()
        # self.lstm = nn.LSTM(input_shape, hidden_shape, batch_first=True)
        self.dense = nn.Sequential(
            nn.Linear(input_shape, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_shape)
        )

    def forward(self, x):
        out = self.dense(x)
        return out


class FacialLandmarksOpenFace(Dataset):

    def __init__(self, paths, transform=None):
        self.paths = []
        for path in paths:
            self.paths += glob.glob(path + "run_*/openface/video.csv")
        self.annotations = pd.read_csv("BagOfLies/Annotations.csv")
        self.vid_annotations = self.annotations["video"].str.replace("./Finalised", "BagOfLies/Finalised").str.replace("/video.mp4", "")
        self.transform = transform
        self.max_seq_len = 1265
        cols = pd.read_csv(self.paths[0]).columns
        self.start_col = np.where(cols == " X_0")[0][0]
        self.end_col = np.where(cols == " Z_67")[0][0]

    def load_expressions(self, file_path):
        df = pd.read_csv(file_path)
        df = df.iloc[:, self.start_col:self.end_col+1]
        df = np.array(df)
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
        df = np.concatenate((mean, std, maximum, minimum, median, var, per_25, per_50, per_75, kurtosis, skew), axis=0)
        df = np.nan_to_num(df, 0)
        df = (df - df.mean(axis = 0) / (df.std(axis = 0)))
        return torch.tensor(df).to(device)

    def get_ground_truth(self, file_path):
        video_run = "/".join(file_path.split("/")[:-2])
        ind = self.vid_annotations[self.vid_annotations == video_run].index
        return self.annotations.iloc[ind]["truth"].values[0]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        X = self.load_expressions(self.paths[index]).to(torch.float32)
        y = torch.tensor(self.get_ground_truth(self.paths[index])).to(device)
        if self.transform:
            return self.transform(X), y
        else:
            return X, y  # return data, label (X, y)
paths = glob.glob("BagOfLies/Finalised/User_*/")
paths.sort()
i = int(sys.argv[1])
train_paths = paths[:i] + paths[i+1:]
test_paths = [paths[i]]

train_dataset = FacialLandmarksOpenFace(train_paths)
test_dataset = FacialLandmarksOpenFace(test_paths)

train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=True)
NUM_EPOCHS = 10
loss_fn = nn.CrossEntropyLoss()
model = NN(2244, 2,2).to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
start_time = timer()

model_results = train(model=model,
                      train_dataloader=train_dataloader,
                      test_dataloader=test_dataloader,
                      optimizer=optimizer,
                      loss_fn=loss_fn,
                      epochs=NUM_EPOCHS)

# End the timer and print out how long it took
end_time = timer()
print(f"Total training time: {end_time-start_time:.3f} seconds")

plot_loss_curves(model_results)

with open("OFLandmarksNN.txt", "a") as file:
    file.write(str(model_results["test_acc"][-1]))