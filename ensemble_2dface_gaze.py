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
import numpy as np
from sklearn.preprocessing import LabelEncoder
import random 
from sklearn.ensemble import GradientBoostingClassifier

def fit_multiple_estimators(classifiers, X_list, y, sample_weights = None):

    # Convert the labels `y` using LabelEncoder, because the predict method is using index-based pointers
    # which will be converted back to original data later.
    le_ = LabelEncoder()
    le_.fit(y)
    transformed_y = le_.transform(y)

    # Fit all estimators with their respective feature arrays
    estimators_ = [clf.fit(X, y) if sample_weights is None else clf.fit(X, y, sample_weights) for clf, X in zip([clf for _, clf in classifiers], X_list)]

    return estimators_, le_


def predict_from_multiple_estimator(estimators, label_encoder, X_list, weights = None):

    # Predict 'soft' voting with probabilities

    pred1 = np.asarray([clf.predict_proba(X) for clf, X in zip(estimators, X_list)])
    pred2 = np.average(pred1, axis=0, weights=weights)
    pred = np.argmax(pred2, axis=1)

    # Convert integer predictions to original labels:
    return label_encoder.inverse_transform(pred)
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
        df = np.save( "/".join(file_path.split('/')[:-1]) + "/of_landmarks_pro.npy", df)
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

# random_skip = random.randint(0, len(train_paths))
# val_path = [train_paths[random_skip]]
# del train_paths[random_skip]

def get_X_y(paths, dataset, start, end):
    ds = dataset(paths, start, end)
    X_r = []
    y_r = []
    for idx in range(len(ds)):
        X, y = ds[idx]
        X_r.append(X.numpy())
        y_r.append(y)
    return np.array(X_r), np.array(y_r)


X_train_face_2d , y_train_face_2d = get_X_y(train_paths, FacialLandmarksOpenFace, " x_0", " y_67")
X_test_face_2d, y_test_face_2d = get_X_y(test_paths, FacialLandmarksOpenFace, " x_0", " y_67")
# X_val_face_2d, y_val_face_2d = get_X_y(val_path, FacialLandmarksOpenFace, " x_0", " y_67")



X_train_gaze, y_train_gaze = get_X_y(train_paths, FacialLandmarksOpenFace, " gaze_0_x", " gaze_1_z")
X_test_gaze, y_test_gaze = get_X_y(test_paths, FacialLandmarksOpenFace, " gaze_0_x", " gaze_1_z")
# X_val_au, y_val_au = get_X_y(val_path, FacialLandmarksOpenFace, " AU01_c", " AU45_c")

X_train_list = [X_train_face_2d, X_train_gaze]
X_test_list = [X_test_face_2d, X_test_gaze]
# X_val_list = [X_val_face_2d, X_val_au]

classifiers = [('svc_face', SVC(probability=True)), ('svc_au', GradientBoostingClassifier())]
fitted_estimators, label_encoder = fit_multiple_estimators(classifiers, X_train_list, y_train_face_2d)
all_weights = []
accs_check = []

# for i in np.arange(0.01, 1, 0.01):
#     weights = [i, 1-i]
#     y_pred_check = predict_from_multiple_estimator(fitted_estimators, label_encoder, X_val_list, weights=weights)
#     all_weights.append(weights)

#     acc = accuracy_score(y_val_face_2d, y_pred_check)
#     accs_check.append(acc)

# max_train_acc_indx = np.argmax(accs_check)
# y_pred_test = predict_from_multiple_estimator(fitted_estimators, label_encoder, X_test_list, weights=all_weights[max_train_acc_indx])
# test_acc = accuracy_score(y_test_face_2d, y_pred_test)
# print(f"Val acc: {max(accs_check)}")
# print(f"Test acc: {test_acc}")

weights = [0.5, 0.5]
y_pred = predict_from_multiple_estimator(fitted_estimators, label_encoder, X_test_list, weights=weights)
test_acc = accuracy_score(y_test_face_2d, y_pred)
with open("accuracies/EnsembleVotingMeanFacial2D_Gaze.txt", "a") as file:
    file.write(str(test_acc) + '\n')

