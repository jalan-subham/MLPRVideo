import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import glob
import pandas as pd
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from utils import *
import warnings
import numpy as np
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import sys 

warnings.filterwarnings("ignore")
device = "cuda:1" if torch.cuda.is_available() else "cpu"
print("Using", device)


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
        self.start_col = np.where(cols == " AU01_c")[0][0]
        self.end_col = np.where(cols == " AU45_c")[0][0]

    def load_expressions(self, file_path):
        df = pd.read_csv(file_path)
        df = df.iloc[:, self.start_col:self.end_col+1]
        df = np.array(df)
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


class AUPresenceLSTM(nn.Module):
    def __init__(self, input_shape, hidden_shape, output_shape):
        super().__init__()
        self.lstm = nn.GRU(input_shape, hidden_shape, batch_first=True)
        self.dense = nn.Sequential(
            nn.Linear(hidden_shape, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, output_shape)
        )

    def forward(self, x, lengths):
        packed_input = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        last_output = output[:, -1, :]  # Get the last output from each sequence
        out = self.dense(last_output)
        # out = last_output
        return out


def collate_pad(batch):
    """Pads sequences to the maximum length within the batch."""
    X_batch, y_batch = zip(*batch)
    lengths = [len(seq) for seq in X_batch]
    X_padded = pad_sequence(X_batch, batch_first=True, padding_value=0)
    return X_padded, lengths, torch.tensor(y_batch)


paths = glob.glob("BagOfLies/Finalised/User_*/")
paths.sort()
i = int(sys.argv[1])
import random 
random.shuffle(paths)
# train_paths = paths[:i] + paths[i+1:]
# test_paths = [paths[i]]

train_paths = paths[:int(len(paths)*0.8)]
test_paths = paths[int(len(paths)*0.8):]
train_dataset = AUPresence(train_paths)
test_dataset = AUPresence(test_paths)

print(f"Training set size: {len(train_dataset)}")
print(f"Validation set size: {len(test_dataset)}")
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_pad, num_workers=2)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True, collate_fn=collate_pad, num_workers=2)

X_batch, lengths, y_batch = next(iter(train_dataloader))
print("Sample Instance:")
print(X_batch)
print(f"X_batch shape: {X_batch.shape}")
print(f"y_batch shape: {y_batch.shape}")

model = AUPresenceLSTM(18, 256, 2).to(device)

NUM_EPOCHS = 50
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
# model = nn.DataParallel(model) 

start_time = timer()

# Train model
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
plt.savefig('outputs/AUPresenceLSTM' + str(i))
# with open('AUPresenceLSTM_mean.txt', 'a') as f:
#     f.write(str(model_results['test_acc'][-1]) + '\n')