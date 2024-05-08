import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2 
import glob 
import warnings
warnings.filterwarnings("ignore")
model_path = 'face_landmarker.task'
import math
import numpy as np 
paths = glob.glob("BagOfLies/Finalised/User_*/run_*/video.mp4")
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    output_face_blendshapes=True)


with FaceLandmarker.create_from_options(options) as landmarker:
    for path in paths:
        print(f"Processing: {path}")
        landmarks = []
        blendshapes = []
        curr = []
        cap = cv2.VideoCapture(path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            timestamp = cv2.CAP_PROP_POS_MSEC
            print(timestamp)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            face_landmarker_result = landmarker.detect_for_video(mp_image, timestamp)
            if face_landmarker_result:
                for face_landmarks in face_landmarker_result.face_landmarks:
                    for lam in face_landmarks:
                        landmarks.extend([lam.x, lam.y, lam.z])
                for blendshape in face_landmarker_result.face_blendshapes:
                    blendshapes.append([x.score for x in blendshape])
        landmarks = np.array(landmarks)

        blendshapes = np.array(blendshapes)
        print(f"Landmarks: {landmarks.shape}")
        print(f"blendshapes: {blendshapes.shape}")
        cap.release()