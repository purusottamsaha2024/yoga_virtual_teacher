import os
from keras.models import load_model
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import math
from gtts import gTTS
import os
from playsound import playsound

from calc_angles import rangles  # Import angle calculation
from recommendations import check_pose_angle  # Import recommendation system



# Configuration
# path_saved_model = "/Users/abcom/Documents/CustomPose-Classification-Mediapipe/model.h5"
path_saved_model = "model.h5"
threshold = 0.5 # Adjust based on your requirement
save = False  # Set to True if you want to save the output video

##############
torso_size_multiplier = 2.5
n_landmarks = 33
n_dimensions = 3
landmark_names = [
    'nose',
    'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist',
    'left_hip', 'right_hip',
    'left_knee', 'right_knee',
    'left_ankle', 'right_ankle',
    'left_heel', 'right_heel',
    'left_foot_index', 'right_foot_index',
]
class_names = [
    'Chair', 'Cobra', 'Dog',
    'Tree', 'Warrior'
]
##############

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

col_names = []
for i in range(n_landmarks):
    name = mp_pose.PoseLandmark(i).name
    name_x = name + '_X'
    name_x = name_x.lower()
    print('name_x:',name_x)
    name_y = name + '_Y'
    name_y = name_y.lower()
    name_z = name + '_Z'
    name_z = name_z.lower()
    name_v = name + '_V'
    name_v = name_v.lower()
    col_names.append(name_x)
    col_names.append(name_y)
    col_names.append(name_z)
    col_names.append(name_v)

# Load saved model
model = load_model(path_saved_model, compile=True)

# # Web-cam
# source = 0  # Default webcam

# cap = cv2.VideoCapture(source)
# source_width = int(cap.get(3))
# source_height = int(cap.get(4))

# # Write Video
# if save:
#     out_video = cv2.VideoWriter('output.avi', 
#                         cv2.VideoWriter_fourcc(*'MJPG'),
#                         10, (source_width, source_height))
    
# Function to load reference landmarks dynamically
def load_reference_landmarks(pose_class):
    # Assuming you have a CSV file with columns for each landmark (e.g., 'left_shoulder_X', 'left_shoulder_Y')
    # reference_df = pd.read_csv("/Users/abcom/Documents/CustomPose-Classification-Mediapipe/landmarks.csv")
    reference_df = pd.read_csv("data.csv")
    
    # Filter the reference landmarks for the predicted pose
    reference_landmarks = reference_df[reference_df['Pose_Class'] == pose_class]
    
    if reference_landmarks.empty:
        return None 
    
    reference_points = {
        'left_shoulder': (int(reference_landmarks['left_shoulder_x'].values[0]),
                          int(reference_landmarks['left_shoulder_y'].values[0])),
        'left_elbow': (int(reference_landmarks['left_elbow_x'].values[0]),
                       int(reference_landmarks['left_elbow_y'].values[0])),
        'left_wrist': (int(reference_landmarks['left_wrist_x'].values[0]),
                       int(reference_landmarks['left_wrist_y'].values[0])),
        # Add other relevant points here
    }
    
    return reference_points  

# def speak_text(text):
#     tts = gTTS(text=text, lang='en')
#     tts.save("output.mp3")
#     playsound("output.mp3")
#     os.remove("output.mp3")  # Optionally, remove the file after playing 

def draw_pose_lines(img, detected_points, reference_points):
    """
    Draws lines on the image between detected and reference points.
    """
    # Colors for detected and reference pose lines
    detected_color = (0, 255, 0)  # Green for detected pose
    reference_color = (0, 0, 255)  # Red for reference pose

    # Check if both detected points and reference points are available
    if detected_points and reference_points:
        for point in detected_points.keys():
            if point in reference_points:
                # Draw line for detected pose
                cv2.line(img, detected_points[point], detected_points[point], detected_color, 2)

                # Draw line for reference pose
                cv2.line(img, reference_points[point], reference_points[point], reference_color, 2)
    return img

def pose_detection():
    cap = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()
        if not success:
            print('[ERROR] Failed to Read Video feed')
            break
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = pose.process(img_rgb)

        if result.pose_landmarks:
            lm_list = []
            for landmarks in result.pose_landmarks.landmark:
                # Preprocessing
                max_distance = 0
                lm_list.append(landmarks)
            center_x = (lm_list[landmark_names.index('right_hip')].x +
                        lm_list[landmark_names.index('left_hip')].x) * 0.5
            center_y = (lm_list[landmark_names.index('right_hip')].y +
                        lm_list[landmark_names.index('left_hip')].y) * 0.5

            shoulders_x = (lm_list[landmark_names.index('right_shoulder')].x +
                        lm_list[landmark_names.index('left_shoulder')].x) * 0.5
            shoulders_y = (lm_list[landmark_names.index('right_shoulder')].y +
                        lm_list[landmark_names.index('left_shoulder')].y) * 0.5

            for lm in lm_list:
                distance = math.sqrt((lm.x - center_x) ** 2 + (lm.y - center_y) ** 2)
                if(distance > max_distance):
                    max_distance = distance
            torso_size = math.sqrt((shoulders_x - center_x) ** 2 +
                                (shoulders_y - center_y) ** 2)
            max_distance = max(torso_size * torso_size_multiplier, max_distance)

            pre_lm = list(np.array([[(landmark.x - center_x) / max_distance, 
                                    (landmark.y - center_y) / max_distance,
                                    landmark.z / max_distance, landmark.visibility] 
                                    for landmark in lm_list]).flatten())
            data = pd.DataFrame([pre_lm], columns=col_names)

            # # Calculate angles
            # landmarks_points = {}  # Dictionary to store landmark points (needed for angle calculations)
            # angles = rangles(data, landmarks_points)  # Compute angles from landmarks

            # angles_df = pd.read_csv("/Users/abcom/Documents/CustomPose-Classification-Mediapipe/csv_files/poses_angles.csv")
            # data = pd.concat([data, angles_df], axis=1)  # Combine landmarks and angles

            predict = model.predict(data)[0]
            print("predict:",predict)
            if max(predict) > threshold:
                pose_class = class_names[predict.argmax()]
                print('predictions: ', predict)
                print('predicted Pose Class: ', pose_class)
                print('predicted Pose Class: ', pose_class)
                # speak_text(f"You are in {pose_class} pose.")  # Give audio feedback for detected pose

                reference_pose_points = load_reference_landmarks(pose_class)

                # Extract detected points
                detected_pose_points = {
                    'left_shoulder': (int(lm_list[landmark_names.index('left_shoulder')].x * img.shape[1]),
                                    int(lm_list[landmark_names.index('left_shoulder')].y * img.shape[0])),
                    'left_elbow': (int(lm_list[landmark_names.index('left_elbow')].x * img.shape[1]),
                                int(lm_list[landmark_names.index('left_elbow')].y * img.shape[0])),
                    'left_wrist': (int(lm_list[landmark_names.index('left_wrist')].x * img.shape[1]),
                                int(lm_list[landmark_names.index('left_wrist')].y * img.shape[0])),
                    # Add other detected points similarly
                }

                # Draw landmarks and connections on the image
                mp_drawing.draw_landmarks(
                    img, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
                # Draw the lines for detected and reference poses
                # Draw the lines for detected and reference poses
                img = draw_pose_lines(img, detected_pose_points, reference_pose_points)

                # Angle calculation
                landmarks_points = {}
                angles = rangles(data, landmarks_points)  # Call the angle calculation function

                # Get corrections based on the predicted pose and angles
                pose_index = class_names.index(pose_class)  # Find the pose index in your dataset
                print("pose_index:",pose_index)
                angles_df = pd.read_csv("refined_pose_angles_with_tolerance.csv")
                # angles_df = pd.read_csv("/Users/abcom/Documents/Yoga_Pose_Classification/refined_pose_angles_with_tolerance.csv")
                # Ensure you fetch the correct row from the DataFrame
                print("angles_df:",angles_df)
                reference_angles = angles_df[angles_df['Pose_Class'] == pose_class]
                reference_angles = reference_angles.to_dict(orient='records')[0]
                print("reference_angles:",reference_angles)
                reference_angles_df = pd.DataFrame([reference_angles])
                corrections = check_pose_angle(pose_class, angles, reference_angles_df)  # Get the corrections

                # Display corrections
                # for idx, correction in enumerate(corrections):
                #     if correction:
                #         img = cv2.putText(img, correction, (40, 80 + idx*30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

                if corrections:
                    correction_message = corrections[0]
                    img = cv2.putText(img, corrections[0], (40, 80), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                    # speak_text(correction_message)
            else:
                pose_class = 'Unknown Pose'
                print('[INFO] Predictions is below given Confidence!!')

            # # Draw landmarks and connections on the image
            # mp_drawing.draw_landmarks(
            #     img, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            #     landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            # )
            # # Draw the lines for detected and reference poses
            # # Draw the lines for detected and reference poses
            # img = draw_pose_lines(img, detected_pose_points, reference_pose_points)

            

            # Show Result
            img = cv2.putText(
                img, f'{pose_class}',
                (40, 50), cv2.FONT_HERSHEY_PLAIN,
                2, (255, 0, 255), 2
            )
        ret, buffer = cv2.imencode('.jpg', img)
        img = buffer.tobytes()

        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')  
    cap.release()  
    #     # Write Video
    #     if save:
    #         out_video.write(img)

    #     cv2.imshow('Output Image', img)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    # cap.release()
    # if save:
    #     out_video.release()
    #     print("[INFO] Out video Saved as 'output.avi'")
    # cv2.destroyAllWindows()
    # print('[INFO] Inference on Videostream is Ended...')
