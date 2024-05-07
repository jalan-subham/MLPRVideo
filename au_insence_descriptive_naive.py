import torch
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import glob 
import sys
import scipy
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings("ignore")
class AUPresence(Dataset):

    def __init__(self, paths, transform=None):
        self.paths = []
        for path in paths:
            self.paths += glob.glob(path + "run_*/openface/video.csv")
        self.annotations = pd.read_csv("BagOfLies/Annotations.csv")
        self.vid_annotations = self.annotations["video"].str.replace("./Finalised", "BagOfLies/Finalised").str.replace("/video.mp4", "")
        self.transform = transform
        self.max_seq_len = 1265
        cols = pd.read_csv(self.paths[0]).columns
        self.start_col = np.where(cols == " AU01_r")[0][0]
        self.end_col = np.where(cols == " AU45_c")[0][0]

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
        return torch.tensor(df)

    def get_ground_truth(self, file_path):
        video_run = "/".join(file_path.split("/")[:-2])
        ind = self.vid_annotations[self.vid_annotations == video_run].index
        return self.annotations.iloc[ind]["truth"].values[0]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        X = self.load_expressions(self.paths[index]).to(torch.float32)
        y = self.get_ground_truth(self.paths[index])
        if self.transform:
            return self.transform(X), y
        else:
            return X, y  # return data, label (X, y)

class GaussianNaiveBayes:
    def __init__(self):
        self.model = GaussianNB()

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

# Data Preparation
paths = glob.glob("BagOfLies/Finalised/User_*/")
paths.sort()
i = int(sys.argv[1])
train_paths = paths[:i] + paths[i+1:]
test_paths = [paths[i]]

train_dataset = AUPresence(train_paths)
test_dataset = AUPresence(test_paths)

# Extracting Features and Labels
X_train = []
y_train = []
for idx in range(len(train_dataset)):
    X, y = train_dataset[idx]
    X_train.append(X.numpy())
    y_train.append(y)
X_train = np.array(X_train)
y_train = np.array(y_train)

X_test = []
y_test = []
for idx in range(len(test_dataset)):
    X, y = test_dataset[idx]
    X_test.append(X.numpy())
    y_test.append(y)
X_test = np.array(X_test)
y_test = np.array(y_test)

# Model Training
model = GaussianNaiveBayes()
model.fit(X_train, y_train)

# Prediction and Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
with open("AUInsenceGNB", "a") as file:
    file.write(f"{accuracy}\n")