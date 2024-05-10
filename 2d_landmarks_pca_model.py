import sklearn
import sys 
import glob 
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
# import tensorflow as tf 

accs = []
accs_tf = []
f1s = []
for i in range(35):
    user_paths = glob.glob("BagOfLies/Finalised/User_*/")

    train_paths_user = user_paths[:i] + user_paths[i+1:]
    test_paths_user = [user_paths[i]]
    train_paths = []
    test_paths = []

    for path in train_paths_user:
        train_paths += glob.glob(path + "/run_*/openface/video.csv")
    for path in test_paths_user:
        test_paths += glob.glob(path + "/run_*/openface/video.csv")

    data = pd.read_csv("gaze.csv")

    data.index = data["paths"]

    X_train = data.loc[train_paths].drop(columns=["truth", "paths"])
    y_train = data.loc[train_paths]["truth"]
    # print(X_train.shape)
    # print(y_train.shape)
    X_test = data.loc[test_paths].drop(columns=["truth", "paths"])
    y_test = data.loc[test_paths]["truth"]
    # print(test_paths)
    # print(X_test.shape)
    # print(y_test.shape)
    # model = SVC(kernel="sigmoid")
    model = RandomForestClassifier()
    # model = GradientBoostingClassifier()
    # model = SVC(kernel="poly")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
    f1 = sklearn.metrics.f1_score(y_test, y_pred)
    print(accuracy)
    print(f1)
    f1s.append(f1)
    
    accs.append(accuracy)

    # model = tf.keras.Sequential([
    #     tf.keras.layers.Dense(11, activation="relu"),
    #     tf.keras.layers.Dense(44, activation="relu"),
    #     tf.keras.layers.Dense(1, activation="sigmoid")
    # ])
    # model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    # model.fit(X_train, y_train, epochs=20, verbose=0)
    # _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    # print("tf model", accuracy)
    # accs_tf.append(accuracy)

print(f"Mean: {sum(accs)/len(accs)}")
print(f"F1 Mean: {sum(f1s)/len(f1s)}")
# print(f"Mean: {sum(accs_tf)/len(accs_tf)}")



