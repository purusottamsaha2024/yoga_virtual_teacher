import numpy as np
import pandas as pd
import mediapipe as mp
import cv2
import os

# Initialize dictionary for storing landmarks
landmarks_list = {
    "left_shoulder": [], "right_shoulder": [],
    "left_elbow": [], "right_elbow": [],
    "left_wrist": [], "right_wrist": [],
    "left_hip": [], "right_hip": [],
    "left_knee": [], "right_knee": [],
    "left_ankle": [], "right_ankle": [],
    "left_heel": [], "right_heel": [],
    "left_foot_index": [], "right_foot_index": [],
}

# Function to calculate angles between three points
def angle(p1, p2, p3):
    a = np.array([p1[0], p1[1]])
    b = np.array([p2[0], p2[1]])
    c = np.array([p3[0], p3[1]])

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
        np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180:
        angle = 360 - angle

    return angle

# Mediapipe pose detection
mp_pose = mp.solutions.pose

# Define landmarks
landmarks = {
    "nose": 0,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16,
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
    "left_heel": 29, "right_heel": 30,
    "left_foot_index": 31, "right_foot_index": 32,
}

# Prepare column names
col_names = []
for i in range(len(landmarks.keys())):
    name = list(landmarks.keys())[i]
    col_names.append(name + "_x")
    col_names.append(name + "_y")
    col_names.append(name + "_z")
    col_names.append(name + "_v")

pose_name = col_names.copy()
pose_name.append("Pose_Class")  # Adding Pose_Class column

# Path to your dataset directory
main_dir = "./poses_dataset/angles"
pose_dir_list = os.listdir(main_dir)
pose_list = []

# CSV file to store angles and pose class
file = open("./csv_files/poses_angles1.csv", "w")
file.write(
    "Pose_Class,armpit_left,armpit_right,elbow_left,elbow_right,hip_left,hip_right,knee_left,knee_right,ankle_left,ankle_right\n")

for i in range(0, len(pose_dir_list)):
    images_dir_list = os.listdir(f"{main_dir}/{pose_dir_list[i]}")
    for l in range(0, len(images_dir_list)):
        pre_list = []
        with mp_pose.Pose(static_image_mode=True, enable_segmentation=True) as pose:
            image = cv2.imread(
                f"{main_dir}/{pose_dir_list[i]}/{images_dir_list[l]}")
            result = pose.process(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            try:
                predict = True
                for landmarks in result.pose_landmarks.landmark:
                    pre_list.append(landmarks)
            except AttributeError:
                print(
                    f"No points {main_dir}/{pose_dir_list[i]}/{images_dir_list[l]}")
                predict = False

        if predict == True:
            gen1116 = np.array([
                [
                    pre_list[m].x,
                    pre_list[m].y,
                    pre_list[m].z,
                    pre_list[m].visibility
                ] for m in range(11, 17)
            ]).flatten().tolist()

            gen2333 = np.array([
                [
                    pre_list[m].x,
                    pre_list[m].y,
                    pre_list[m].z,
                    pre_list[m].visibility
                ] for m in range(23, 33)
            ]).flatten().tolist()

            gen1116.extend(gen2333)

            all_list = [
                pre_list[0].x,
                pre_list[0].y,
                pre_list[0].z,
                pre_list[0].visibility,
            ]

            all_list.extend(gen1116)
            tpl = all_list.copy()
            tpl.append(pose_dir_list[i])  # Append pose class here
            pose_list.append(tpl)
data_pose = pd.DataFrame(pose_list, columns=pose_name)

for i, row in data_pose.iterrows():
    sl = []
    landmarks_list["left_shoulder"] = [
        row["left_shoulder_x"], row["left_shoulder_y"]]

    landmarks_list["right_shoulder"] = [
        row["right_shoulder_x"], row["right_shoulder_y"]]

    landmarks_list["left_elbow"] = [
        row["left_elbow_x"], row["left_elbow_y"]]

    landmarks_list["right_elbow"] = [
        row["right_elbow_x"], row["right_elbow_y"]]

    landmarks_list["left_wrist"] = [
        row["left_wrist_x"], row["left_wrist_y"]]

    landmarks_list["right_wrist"] = [
        row["right_wrist_x"], row["right_wrist_y"]]

    landmarks_list["left_hip"] = [
        row["left_hip_x"], row["left_hip_y"]]

    landmarks_list["right_hip"] = [
        row["right_hip_x"], row["right_hip_y"]]

    landmarks_list["left_knee"] = [
        row["left_knee_x"], row["left_knee_y"]]

    landmarks_list["right_knee"] = [
        row["right_knee_x"], row["right_knee_y"]]

    landmarks_list["left_ankle"] = [
        row["left_ankle_x"], row["left_ankle_y"]]

    landmarks_list["right_ankle"] = [
        row["right_ankle_x"], row["right_ankle_y"]]

    landmarks_list["left_heel"] = [
        row["left_heel_x"], row["left_heel_y"]]

    landmarks_list["right_heel"] = [
        row["right_heel_x"], row["right_heel_y"]]

    landmarks_list["left_foot_index"] = [
        row["left_foot_index_x"], row["left_foot_index_y"]]

    landmarks_list["right_foot_index"] = [
        row["right_foot_index_x"], row["right_foot_index_y"]]

    armpit_left = angle(
        landmarks_list["left_elbow"],
        landmarks_list["left_shoulder"],
        landmarks_list["left_hip"]
    )
    armpit_right = angle(
        landmarks_list["right_elbow"],
        landmarks_list["right_shoulder"],
        landmarks_list["right_hip"]
    )

    elbow_left = angle(
        landmarks_list["left_shoulder"],
        landmarks_list["left_elbow"],
        landmarks_list["left_wrist"]
    )
    elbow_right = angle(
        landmarks_list["right_shoulder"],
        landmarks_list["right_elbow"],
        landmarks_list["right_wrist"]
    )

    hip_left = angle(
        landmarks_list["right_hip"],
        landmarks_list["left_hip"],
        landmarks_list["left_knee"]
    )
    hip_right = angle(
        landmarks_list["left_hip"],
        landmarks_list["right_hip"],
        landmarks_list["right_knee"]
    )

    knee_left = angle(
        landmarks_list["left_hip"],
        landmarks_list["left_knee"],
        landmarks_list["left_ankle"]
    )
    knee_right = angle(
        landmarks_list["right_hip"],
        landmarks_list["right_knee"],
        landmarks_list["right_ankle"]
    )

    ankle_left = angle(
        landmarks_list["left_knee"],
        landmarks_list["left_ankle"],
        landmarks_list["left_foot_index"]
    )
    ankle_right = angle(
        landmarks_list["right_knee"],
        landmarks_list["right_ankle"],
        landmarks_list["right_foot_index"]
    )

    tmp = [
        armpit_left,
        armpit_right,
        elbow_left,
        elbow_right,
        hip_left,
        hip_right,
        knee_left,
        knee_right,
        ankle_left,
        ankle_right
    ]

    sl.append(tmp)
    file.write(
        f"{row['Pose_Class']},{','.join(map(lambda x: str(int(round(x))), sl[0]))}\n"
    )
file.close()
