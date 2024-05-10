import glob 
import sklearn 
import pandas as pd
import numpy as np 
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import re
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import sys 

def remove_non_alphanumeric(text):
    return re.sub(r'[^a-zA-Z\s]', '', text)

def remove_comments(text):
    return re.sub(r'(?s)<!--.*?-->', '', text)

def preprocess_text(text):
    text = remove_comments(text)
    text = remove_non_alphanumeric(text)
    return text

def get_bow_features(train_transcript_paths, test_transcript_paths):
    train_transcripts = []
    train_labels = []
    for path in train_transcript_paths:
        with open(path, 'r', encoding='utf-8') as file:
            transcript = file.read()
            preprocessed_transcript = preprocess_text(transcript)
            train_transcripts.append(preprocessed_transcript)
            # label = path.split('/')[-1].split('_')[0]
            # train_labels.append(label)

    test_transcripts = []
    test_labels = []
    for path in test_transcript_paths:
        with open(path, 'r', encoding='utf-8') as file:
            transcript = file.read()
            preprocessed_transcript = preprocess_text(transcript)
            test_transcripts.append(preprocessed_transcript)
            # label = path.split('/')[-1].split('_')[0]
            # test_labels.append(label)

    vectorizer = CountVectorizer()

    X_train = vectorizer.fit_transform(train_transcripts).toarray()

    X_test = vectorizer.transform(test_transcripts).toarray()
    
    return X_train, X_test

def get_gaze_data(train_paths, test_paths):
    df = pd.read_csv("gaze.csv")
    df.index = df["paths"]
    X_train = df.loc[train_paths].drop(columns=["truth", "paths"])
    y_train = df.loc[train_paths]["truth"]
    X_test = df.loc[test_paths].drop(columns=["truth", "paths"])
    y_test = df.loc[test_paths]["truth"]
    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

def get_vid_data(train_paths, test_paths):
    df = pd.read_csv("2d_landmarks_pca_setb.csv")
    df.index = df["paths"]
    X_train = df.loc[train_paths].drop(columns=["truth", "paths"])
    y_train = df.loc[train_paths]["truth"]
    X_test = df.loc[test_paths].drop(columns=["truth", "paths"])
    y_test = df.loc[test_paths]["truth"]
    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

def get_audio_data(train_paths, test_paths):
    df = pd.read_csv("audio_features_set_B (3).csv")
    users = df["usernum"].tolist()
    # runs = []
    # i = 1
    # last = users[0]
    # runs.append(0)
    # for user in users[1:]:
    #     if user != last:
    #         i = 0
    #         last = user 
    #     runs.append(i)
    #     i += 1
    # df["runs"] = runs 
    df.index = "BagOfLies/Finalised/User_" + df["usernum"].astype(str) + "/run_" + df["run"].astype(str) + "/openface/video.csv" 
    vals = df.index.values
    vals.sort()

    train_paths.sort()


    X_train = df.loc[train_paths].drop(columns=["truth", "usernum", "run"])
    y_train = df.loc[train_paths]["truth"]
    X_test = df.loc[test_paths].drop(columns=["truth", "usernum", "run"])
    y_test = df.loc[test_paths]["truth"]
    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)


def fit_multiple_estimators(classifiers, X_list, y, sample_weights = None):

    
    le_ = LabelEncoder()
    le_.fit(y)
    transformed_y = le_.transform(y)

    estimators_ = [clf.fit(X, y) if sample_weights is None else clf.fit(X, y, sample_weights) for clf, X in zip([clf for _, clf in classifiers], X_list)]

    return estimators_, le_


def predict_from_multiple_estimator(estimators, label_encoder, X_list, weights = None):


    pred1 = np.asarray([clf.predict_proba(X) for clf, X in zip(estimators, X_list)])
    # print(pred1)
    pred2 = np.average(pred1, axis=0, weights=weights)
    # print(pred2)
    pred = np.argmax(pred2, axis=1)

    return label_encoder.inverse_transform(pred)

all_avail = pd.read_csv("setb.csv")

# print(all_avail)
# print(all_avail.groupby("usernum").aggregate({"run":"count"}))
# print(all_avail.groupby("usernum").aggregate({"result":"mean"}))
# print(all_avail["result"].mean())

all_avail["paths"] = "BagOfLies/Finalised/User_" + all_avail["usernum"].astype(str) + "/run_" + all_avail["run"].astype(str) + "/openface/video.csv"
all_avail.index = all_avail["paths"]

balanced_accs = []
accs_mega= []
f1s = []
f1s_mega = []
users = glob.glob("BagOfLies/Finalised/User_*/")

audio, video, gaze, transcript = True, True, True, True
print(audio, video, gaze, transcript)

for i in range(35):
    classifiers = []
    if gaze:
        classifiers.append(("rf",RandomForestClassifier()))
    if video:
        classifiers.append(("rf", SVC(kernel="sigmoid", probability=True)))
        # classifiers.append(("rf", RandomForestClassifier()))
    if audio:
        classifiers.append(("svc", SVC(kernel="rbf", probability=True)))
    if transcript:
        classifiers.append(("rf", RandomForestClassifier()))
    print(f"User {i}")
    train_paths_users = users[:i] + users[i+1:]
    test_paths_users = [users[i]]
    train_paths = []
    test_paths = []
    for path in train_paths_users:
        tentative =  glob.glob(path + "/run_*/openface/video.csv")
        train_paths += [x for x in tentative if x in all_avail["paths"].values]
    for path in test_paths_users:
        tentative =  glob.glob(path + "/run_*/openface/video.csv")
        test_paths += [x for x in tentative if x in all_avail["paths"].values]
    X_train_gaze, y_train_gaze, X_test_gaze, y_test_gaze = get_gaze_data(train_paths, test_paths)
    X_train_vid, y_train_vid, X_test_vid, y_test_vid = get_vid_data(train_paths, test_paths)
    X_train_audio, y_train_audio, X_test_audio, y_test_audio = get_audio_data(train_paths, test_paths)
    X_train_trans, X_test_trans = get_bow_features([
        x.replace("openface/video.csv", "transcript.txt") for x in train_paths
    ], [x.replace("openface/video.csv", "transcript.txt") for x in test_paths])
    X_list = []
    X_test = []
    # print(X_test_audio.shape, X_test_gaze.shape, X_test_vid.shape, X_test_trans.shape)
    if gaze:
        X_list.append(X_train_gaze)
        X_test.append(X_test_gaze)
    if video:
        X_list.append(X_train_vid)
        X_test.append(X_test_vid)
    if audio:
        X_list.append(X_train_audio)
        X_test.append(X_test_audio)
    if transcript:
        X_list.append(X_train_trans)
        X_test.append(X_test_trans)
    y = y_train_gaze
    estimators, le = fit_multiple_estimators(classifiers, X_list, y)
    y = y_test_gaze
    y_pred = predict_from_multiple_estimator(estimators, le, X_test)
    balanced_acc = sklearn.metrics.balanced_accuracy_score(y, y_pred)
    balanced_accs.append(balanced_acc)
    f1 = sklearn.metrics.f1_score(y, y_pred)
    f1s.append(f1)

    X_mega_train = np.concatenate(X_list, axis=1)
    X_mega_test = np.concatenate(X_test, axis=1)
    model = SVC()
    model.fit(X_mega_train, y_train_gaze)
    y_pred = model.predict(X_mega_test)
    accuracy = sklearn.metrics.balanced_accuracy_score(y_test_gaze, y_pred)
    f1 = sklearn.metrics.f1_score(y_test_gaze, y_pred)
    accs_mega.append(accuracy)
    f1s_mega.append(f1)

# print(balanced_accs)
print(f"Average accuracy for individual models: {np.mean(balanced_accs)}")
# print(accs_mega)
print(f"Average accuracy for mega model: {np.mean(accs_mega)}") 

print(f"Average f1 for individual models: {np.mean(f1s)}")
print(f"Average f1 for mega model: {np.mean(f1s_mega)}")

with open("logs.txt", "a") as file:
    file.write(f"Modalities: Audio:{audio} Video:{video} Gaze:{gaze} Transcript:{transcript}\n")
    file.write(f"Average accuracy for individual models: {np.mean(balanced_accs)}\n")
    file.write(f"Average accuracy for mega model: {np.mean(accs_mega)}\n")
    file.write(f"Average f1 for individual models: {np.mean(f1s)}\n")
    file.write(f"Average f1 for mega model: {np.mean(f1s_mega)}\n")
