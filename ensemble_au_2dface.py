import torch
import pandas as pd
import numpy as np
from torch import nn, optim
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import glob 
import sys
import scipy
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings("ignore")
from sklearn.svm import SVC
from timeit import default_timer as timer
import utils_basic_sgm
device = "cuda:1"
import os
from sklearn.ensemble import VotingClassifier

os.environ["CUDA_LAUNCH_BLOCKING"] = str(1)
class FacialLandmarksOpenFace(Dataset):

    def __init__(self, paths, start, end, transform=None):
        self.paths = []
        for path in paths:
            self.paths += glob.glob(path + "run_*/openface/video.csv")
        self.annotations = pd.read_csv("BagOfLies/Annotations.csv")
        self.vid_annotations = self.annotations["video"].str.replace("./Finalised", "BagOfLies/Finalised").str.replace("/video.mp4", "")
        self.transform = transform
        self.max_seq_len = 1265
        cols = pd.read_csv(self.paths[0]).columns
        self.start_col = np.where(cols == start)[0][0]
        self.end_col = np.where(cols == end)[0][0]

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

paths = glob.glob("BagOfLies/Finalised/User_*/")
paths.sort()
i = int(sys.argv[1])
train_paths = paths[:i] + paths[i+1:]
test_paths = [paths[i]]

train_ds_face2d = FacialLandmarksOpenFace(train_paths, " x_0", " y_67")
test_ds_face2d = FacialLandmarksOpenFace(test_paths, " x_0", " y_67")
X_train_face_2d = []
y_train_face_2d = []

X_test_face_2d = []
y_test_face_2d = []
for idx in range(len(train_ds_face2d)):
    X, y = train_ds_face2d[idx]
    X_train_face_2d.append(X.numpy())
    y_train_face_2d.append(y)

for idx in range(len(test_ds_face2d)):
    X, y = test_ds_face2d[idx]
    X_test_face_2d.append(X.numpy())
    y_test_face_2d.append(y)

X_train_face_2d = np.array(X_train_face_2d)
y_train_face_2d = np.array(y_train_face_2d)
X_test_face_2d = np.array(X_test_face_2d)
y_test_face_2d = np.array(y_test_face_2d)

svc_face_2d =  SVC(probability=True)
svc_face_2d.fit(X_train_face_2d, y_train_face_2d)
print("Fitted Face 2D Landmarks")

face_2d_probs_train = svc_face_2d.predict_proba(X_train_face_2d)
face_2d_probs_test = svc_face_2d.predict_proba(X_test_face_2d)

print(f"Face2D Train Acc: {(svc_face_2d.predict(X_train_face_2d) == y_train_face_2d).mean()}")
train_ds_au = FacialLandmarksOpenFace(train_paths, " AU01_c", " AU45_c")
test_ds_au = FacialLandmarksOpenFace(test_paths, " AU01_c", " AU45_c")

X_train_au = []
y_train_au = []
X_test_au = []
y_test_au = []
for idx in range(len(train_ds_au)):
    X, y = train_ds_au[idx]
    X_train_au.append(X.numpy())
    y_train_au.append(y)
X_train_au = np.array(X_train_au)
y_train_au = np.array(y_train_au)

for idx in range(len(test_ds_au)):
    X, y = train_ds_au[idx]
    X_test_au.append(X.numpy())
    y_test_au.append(y)

X_test_au = np.array(X_test_au)
y_test_au  = np.array(y_test_au)

svc_au = SVC(probability=True)
svc_au.fit(X_train_au, y_train_au)
print("Fitted AU")

au_probs_train = svc_au.predict_proba(X_train_au)
au_probs_test = svc_au.predict_proba(X_test_au)

print(f"AU Train Acc: {(svc_au.predict(X_train_au) == y_train_au).mean()}")

voting_clf = VotingClassifier(estimators=[('svc1', svc_face_2d), ('svc2', svc_au)], voting='hard')
voting_clf.fit(X_train_face_2d, y_train_face_2d)
voting_clf.fit(X_train_au, y_train_au)


class EnsembleDataset(Dataset):

    def __init__(self, probs, true_labels):
        
        self.probs = torch.tensor(probs).to(torch.float32)
        self.true_labels = torch.tensor(true_labels).to(torch.float32).reshape(-1, 1)
    
    def __len__(self):
        return self.probs.shape[0]

    def __getitem__(self, index):
        return self.probs[index].to(device), self.true_labels[index].to(device)


X_train_ens = np.hstack([au_probs_train[:], face_2d_probs_train[:]])
y_train_ens = y_train_face_2d
ens_ds_train = EnsembleDataset(X_train_ens, y_train_ens)
ens_dataloader_train = DataLoader(ens_ds_train, batch_size=256)

X_test_ens = np.hstack([au_probs_test[:], face_2d_probs_test[:]])
y_test_ens = y_test_face_2d

ens_ds_test = EnsembleDataset(X_test_ens, y_test_ens)
ens_dataloader_test = DataLoader(ens_ds_test, batch_size=32)

class EnsModel(nn.Module):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.dense = nn.Sequential(
            nn.Linear(input_shape, 2),
            nn.ReLU(),
            nn.Linear(2, output_shape),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.dense(x)

ens_model = EnsModel(4, 1).to(device)

optimizer =  optim.Adam(ens_model
                        .parameters(), lr=0.001)
loss_fn =  nn.BCELoss()

NUM_EPOCHS = 10

model_results = utils_basic_sgm.train(
    ens_model, ens_dataloader_train, ens_dataloader_test, optimizer=optimizer, loss_fn=loss_fn, epochs=NUM_EPOCHS
)
with open("accuracies/Ensemble2DFacialAU.txt", "a") as file:
    file.write(str(np.array(model_results["test_acc"]).mean()) + '\n')