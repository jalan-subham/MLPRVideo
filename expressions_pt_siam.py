import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import random
import glob
import warnings 
import tqdm
warnings.filterwarnings("ignore")

device = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 50

cats = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

le = OneHotEncoder(sparse_output=False)
le.fit(np.array(cats).reshape(-1, 1))

# Define Siamese Network architecture
class SiameseGRU(nn.Module):
    def __init__(self, input_shape, hidden_shape, output_shape):
        super(SiameseGRU, self).__init__()
        self.gru = nn.GRU(input_shape, hidden_shape, batch_first=True)
        self.dense = nn.Sequential(
            nn.Linear(hidden_shape, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_shape)
        )

    def forward_once(self, x):
        out, _ = self.gru(x)
        out = self.dense(out[:, -1])
        return out

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2
    
    def get_embeddings(self, x):
        # Method to extract embeddings from input
        return self.forward_once(x)
# Triplet loss function
class MaximizeTripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(MaximizeTripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = torch.pairwise_distance(anchor, positive, p=2)
        distance_negative = torch.pairwise_distance(anchor, negative, p=2)
        losses = torch.relu(distance_negative - distance_positive + self.margin)  # Flipped signs
        return losses.mean()


# Modify the training loop to accommodate triplet loss
def train_maximize_triplet(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for batch in tqdm.tqdm(dataloader):
        anchor, positive, negative = batch
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
        
        optimizer.zero_grad()
        
        anchor_output, positive_output = model(anchor, positive)
        negative_output = model.forward_once(negative)
        
        loss = criterion(anchor_output, positive_output, negative_output)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    return running_loss / len(dataloader)

# Create custom Dataset class for triplet data
class TripletExpressionDataset(Dataset):
    def __init__(self, paths):
        self.paths = []
        for path in paths:
            self.paths += glob.glob(path + "run_*/expressions.txt")
        self.annotations = pd.read_csv("BagOfLies/Annotations.csv")
        self.vid_annotations = self.annotations["video"].str.replace("./Finalised", "BagOfLies/Finalised").str.replace("/video.mp4", "")
        self.max_seq_len = 1265

    def load_expressions(self, file_path):
        df = pd.read_csv(file_path)
        df = le.transform(df)
        return torch.tensor(df).to(torch.float32)
    
    def get_ground_truth(self, file_path):
        video_run = "/".join(file_path.split("/")[:-1])
        ind = self.vid_annotations[self.vid_annotations == video_run].index
        return self.annotations.iloc[ind]["truth"].values[0]
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        anchor_path = self.paths[index]
        anchor_X = self.load_expressions(anchor_path)
        anchor_y = self.get_ground_truth(anchor_path)
        
        # Positive example: randomly select another example from the same class
        positive_class_paths = [p for p in self.paths if self.get_ground_truth(p) == anchor_y]
        positive_path = random.choice(positive_class_paths)
        positive_X = self.load_expressions(positive_path)
        
        # Negative example: randomly select an example from a different class
        negative_class_paths = [p for p in self.paths if self.get_ground_truth(p) != anchor_y]
        negative_path = random.choice(negative_class_paths)
        negative_X = self.load_expressions(negative_path)
        
        return anchor_X, positive_X, negative_X
    
def collate_pad(batch):
    anchor_X = [item[0] for item in batch]
    positive_X = [item[1] for item in batch]
    negative_X = [item[2] for item in batch]
    anchor_X = nn.utils.rnn.pad_sequence(anchor_X, batch_first=True, padding_value=0)
    positive_X = nn.utils.rnn.pad_sequence(positive_X, batch_first=True, padding_value=0)
    negative_X = nn.utils.rnn.pad_sequence(negative_X, batch_first=True, padding_value=0)
    return anchor_X, positive_X, negative_X

paths = glob.glob("BagOfLies/Finalised/User_*/")
train_paths, test_paths = torch.utils.data.random_split(paths, [int(0.8*len(paths)), len(paths) - int(0.8*len(paths))])
# Initialize datasets and dataloaders
triplet_dataset = TripletExpressionDataset(train_paths)
triplet_dataloader = DataLoader(triplet_dataset, batch_size=1, shuffle=True, collate_fn=collate_pad)

# Initialize Siamese LSTM model and other components
siamese_model = SiameseGRU(7, 64, 2).to(device)
maximize_triplet_loss_fn = MaximizeTripletLoss()  # Use the modified triplet loss
triplet_optimizer = optim.Adam(siamese_model.parameters(), lr=0.001)

test_dataset = TripletExpressionDataset(test_paths)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn=collate_pad)
# Train the Siamese model with the modified triplet loss
for epoch in range(NUM_EPOCHS):
    triplet_loss = train_maximize_triplet(siamese_model, triplet_dataloader, triplet_optimizer, maximize_triplet_loss_fn, device)
    print(f"Epoch {epoch+1}, Maximized Triplet Loss: {triplet_loss:.4f}")

# Function to extract embeddings
def extract_embeddings(model, dataloader, device):
    embeddings = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            input_X = batch[0].to(device)
            output = model.get_embeddings(input_X)
            embeddings.append(output.cpu().numpy())
    return np.vstack(embeddings)
torch.save(siamese_model.state_dict(), "siamese_model.pt")
# Extract embeddings from the test dataset
test_embeddings = extract_embeddings(siamese_model, test_dataloader, device)
print("Test embeddings shape:", test_embeddings.shape)