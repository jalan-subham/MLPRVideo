import glob 
import sklearn 
import pandas as pd
import numpy as np 
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

def get_gaze_data(train_paths, test_paths):
    df = pd.read_csv("gaze.csv")
    df.index = df["paths"]
    X_train = df.loc[train_paths].drop(columns=["truth", "paths"])
    y_train = df.loc[train_paths]["truth"]
    X_test = df.loc[test_paths].drop(columns=["truth", "paths"])
    y_test = df.loc[test_paths]["truth"]
    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

def get_vid_data(train_paths, test_paths):
    df = pd.read_csv("2d_landmarks_pca.csv")
    df.index = df["paths"]
    X_train = df.loc[train_paths].drop(columns=["truth", "paths"])
    y_train = df.loc[train_paths]["truth"]
    X_test = df.loc[test_paths].drop(columns=["truth", "paths"])
    y_test = df.loc[test_paths]["truth"]
    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

def get_audio_data(train_paths, test_paths):
    df = pd.read_csv("audio.csv")
    users = df["Usernum"].tolist()
    runs = []
    i = 1
    last = users[0]
    runs.append(0)
    for user in users[1:]:
        if user != last:
            i = 0
            last = user 
        runs.append(i)
        i += 1
    df["runs"] = runs 
    df.index = "BagOfLies/Finalised/User_" + df["Usernum"].astype(str) + "/run_" + df["runs"].astype(str) + "/openface/video.csv" 
    X_train = df.loc[train_paths].drop(columns=["Truth", "Usernum", "runs"])
    y_train = df.loc[train_paths]["Truth"]
    X_test = df.loc[test_paths].drop(columns=["Truth", "Usernum", "runs"])
    y_test = df.loc[test_paths]["Truth"]
    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)


def fit_multiple_estimators(classifiers, X_list, y, sample_weights = None):

    
    le_ = LabelEncoder()
    le_.fit(y)
    transformed_y = le_.transform(y)

    estimators_ = [clf.fit(X, y) if sample_weights is None else clf.fit(X, y, sample_weights) for clf, X in zip([clf for _, clf in classifiers], X_list)]

    return estimators_, le_


def predict_from_multiple_estimator(estimators, label_encoder, X_list, weights = None):


    pred1 = np.asarray([clf.predict_proba(X) for clf, X in zip(estimators, X_list)])
    pred2 = np.average(pred1, axis=0, weights=weights)
    pred = np.argmax(pred2, axis=1)

    return label_encoder.inverse_transform(pred)

users = glob.glob("BagOfLies/Finalised/User_*/")

accs = []
accs_mega= []

audio, video, gaze = True, True, True
for i in range(35):
    classifiers = []
    if gaze:
        classifiers.append(("rf", RandomForestClassifier()))
    if video:
        classifiers.append(("rf", SVC(kernel="sigmoid", probability=True)))
    if audio:
        classifiers.append(("svc", SVC(kernel="rbf", probability=True)))
    print(f"User {i}")
    train_paths_users = users[:i] + users[i+1:]
    test_paths_users = [users[i]]
    train_paths = []
    test_paths = []
    for path in train_paths_users:
        train_paths += glob.glob(path + "/run_*/openface/video.csv")
    for path in test_paths_users:
        test_paths += glob.glob(path + "/run_*/openface/video.csv")
    X_train_gaze, y_train_gaze, X_test_gaze, y_test_gaze = get_gaze_data(train_paths, test_paths)
    X_train_vid, y_train_vid, X_test_vid, y_test_vid = get_vid_data(train_paths, test_paths)
    X_train_audio, y_train_audio, X_test_audio, y_test_audio = get_audio_data(train_paths, test_paths)
    X_list = []
    X_test = []
    if gaze:
        X_list.append(X_train_gaze)
        X_test.append(X_test_gaze)
    if video:
        X_list.append(X_train_vid)
        X_test.append(X_test_vid)
    if audio:
        X_list.append(X_train_audio)
        X_test.append(X_test_audio)
    y = y_train_gaze
    estimators, le = fit_multiple_estimators(classifiers, X_list, y)
    y = y_test_gaze
    y_pred = predict_from_multiple_estimator(estimators, le, X_test)
    accuracy = sklearn.metrics.accuracy_score(y, y_pred)
    print(accuracy)
    accs.append(accuracy)

    X_mega_train = np.concatenate(X_list, axis=1)
    X_mega_test = np.concatenate(X_test, axis=1)
    model = SVC()
    model.fit(X_mega_train, y_train_gaze)
    y_pred = model.predict(X_mega_test)
    accuracy = sklearn.metrics.accuracy_score(y_test_gaze, y_pred)
    accs_mega.append(accuracy)


print(f"Average accuracy for individual models: {np.mean(accs)}")
print(f"Average accuracy for mega model: {np.mean(accs_mega)}") 