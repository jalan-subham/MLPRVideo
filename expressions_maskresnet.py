
import glob
import tqdm 
import cv2
import os
import numpy as np 
files = glob.glob("BagOfLies/Finalised/User_*/run_*/video.mp4")

for file in tqdm.tqdm(files):
    with open(file.replace("video.mp4", "expressions.txt"), "r") as f:
        if f.read():
            continue
    print(f"Parsing {file}...")

    cap = cv2.VideoCapture(file)
    frames = []
    expressions = []
    # Loop through each frame of the video
    for i in tqdm.tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
        # Read the next frame
        ret, frame = cap.read()

        if not ret:
            break

        # Convert the frame to BGR
        # Append the frame to the list
        
    

    # Convert the list of frames to a numpy array
    print(expressions)
    with open(file.replace("video.mp4", "expressions.txt"), "w") as f:
        for expression in expressions:
            f.write(expression + "\n")
    # Release the video file
    cap.release()
