
def init():
    angles_dict = {
        "armpit_left": 0,
        "armpit_right": 1,
        "elbow_left": 2,
        "elbow_right": 3,
        "hip_left": 4,
        "hip_right": 5,
        "knee_left": 6,
        "knee_right": 7,
        "ankle_left": 8,
        "ankle_right": 9,
    }
    return angles_dict

def error_margin(value, min_threshold, max_threshold, margin=40):
    # Apply a margin of ±10 degrees
    adjusted_min = min_threshold - margin
    adjusted_max = max_threshold + margin
    
    print(f"Checking if {value} is within the adjusted range {adjusted_min} to {adjusted_max}")
    
    if adjusted_min <= value <= adjusted_max:
        print(f"{value} is within the acceptable range.")
        return True
    print(f"{value} is outside the acceptable range.")
    return False


def check_joint(angles, joint_name, min_threshold, max_threshold, body_position):
    angles_dict = init()
    joint_index = angles_dict[joint_name]

    print(f"\nChecking joint: {joint_name}")
    print(f"Expected range for {joint_name}: {min_threshold} to {max_threshold}")
    print(f"Calculated angle for {joint_name}: {angles[joint_index]}")

    # Check if the angle is within the acceptable range
    if error_margin(angles[joint_index], min_threshold, max_threshold):
        print(f"{joint_name} is correct, no correction needed.")
        return None

    # Provide feedback based on whether the angle is too large or too small
    if angles[joint_index] > max_threshold:
        print(f"{joint_name} angle is too large. Correction: Move closer.")
        return f"Bring {' '.join(joint_name.split('_')[::-1])} closer to {body_position}."
    elif angles[joint_index] < min_threshold:
        print(f"{joint_name} angle is too small. Correction: Move further away.")
        return f"Put {' '.join(joint_name.split('_')[::-1])} further away from {body_position}."

    return None


def check_pose_angle(pose_class, angles, df):
    corrections = []

    # Directly use the first row, as df is already filtered for the given pose
    reference_row = df.iloc[0]
    print("reference_row:", reference_row)
    print(f"Checking corrections for pose class: {pose_class}")

    joints_to_check = [
        ("armpit_right", "body"),
        ("armpit_left", "body"),
        ("elbow_right", "arm"),
        ("elbow_left", "arm"),
        ("hip_right", "pelvis"),
        ("hip_left", "pelvis"),
        ("knee_right", "calf"),
        ("knee_left", "calf"),
        ("ankle_right", "foot"),
        ("ankle_left", "foot")
    ]

    for joint_name, body_position in joints_to_check:
        # Fetch the corresponding min and max thresholds for the joint
        min_threshold = reference_row[f"{joint_name}_min"]
        max_threshold = reference_row[f"{joint_name}_max"]

        # Check for the correct angle and provide feedback
        joint_correction = check_joint(
            angles, joint_name, min_threshold, max_threshold, body_position
        )
        if joint_correction:
            corrections.append(joint_correction)
            print(f"Correction needed for joint {joint_name}: {joint_correction}")
            return corrections  # Return only the first correction found

    print("No further corrections needed.")
    return None  # Return None when all corrections are done
