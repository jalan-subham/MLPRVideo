import glob 
import pandas as pd 
import numpy as np
import scipy
import tqdm 
from sklearn.decomposition import PCA

paths = glob.glob("BagOfLies/Finalised/User_*/run_*/openface/video.csv")
annotations = pd.read_csv("BagOfLies/Annotations.csv")
vid_annotations = annotations["video"].str.replace("./Finalised", "BagOfLies/Finalised").str.replace("/video.mp4", "")
cols = pd.read_csv(paths[0]).columns
start_col = np.where(cols == " gaze_angle_x")[0][0]
end_col = np.where(cols == " gaze_angle_y")[0][0]
final = []
def get_row(path):
    df = pd.read_csv(path)
    df = df.iloc[:, start_col:end_col+1]
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
    return df.reshape(1, -1)
all = []
truths = []
for path in tqdm.tqdm(paths):
    all.append(get_row(path))
    video_run = "/".join(path.split("/")[:-2])
    ind = vid_annotations[vid_annotations == video_run].index
    truths.append(annotations.iloc[ind]["truth"].values[0])

all = np.array(all)

all = all.reshape(all.shape[0], all.shape[2])
# standardize 
all = (all - np.mean(all, axis=0)) / np.std(all, axis=0)
print(all.shape)



# PCA
# PCA
# pca = PCA(n_components=0.95)
# pca.fit(all)
# all = pca.transform(all)
df = pd.DataFrame(all)
df["truth"] = truths
df["paths"] = paths

df.to_csv("gaze.csv", index=False)

