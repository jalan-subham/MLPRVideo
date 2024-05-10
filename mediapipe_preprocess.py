import cv2
import mediapipe as mp
import glob

# Initialize MediaPipe face detection and landmark modules
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Function to detect landmarks on a frame
def detect_landmarks(frame):
    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(frame)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1),
                )
    return frame

# Paths to videos
paths = glob.glob("BagOfLies/Finalised/User_0/run_0/video.mp4")

# Process and save videos
for path in paths[0:1]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("Error: Could not open video ", path)
        continue 
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('landmarks_processed.mp4', fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process the frame (detect landmarks)
        processed_frame = detect_landmarks(frame)

        # Write the processed frame to the output video
        out.write(processed_frame)

        cv2.imshow('Processed Video', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release everything when done
    cap.release()
    out.release()
    cv2.destroyAllWindows()
