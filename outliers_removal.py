import pandas as pd
import numpy as np

# Set a tolerance range (e.g., ±5% around the mean) and a minimum allowed range.
TOLERANCE_PERCENTAGE = 0.05  # Adjust this for a narrower or wider range
MIN_RANGE = 5  # Minimum angle difference to prevent collapse of min and max

# Calculate a narrow range around the mean with a minimum range constraint
def calculate_narrow_range(data, tolerance=TOLERANCE_PERCENTAGE, min_range=MIN_RANGE):
    mean_value = np.mean(data)
    tolerance_range = mean_value * tolerance
    min_value = max(np.min(data), mean_value - tolerance_range)
    max_value = min(np.max(data), mean_value + tolerance_range)
    
    # Ensure that min and max have at least the minimum allowed range
    if max_value - min_value < min_range:
        max_value = min_value + min_range
    
    return min_value, max_value

# Define a function to process the angle data for each pose and joint
def process_pose_angles_narrow(df, joints, tolerance=TOLERANCE_PERCENTAGE, min_range=MIN_RANGE):
    refined_data = {}
    
    for pose_class in df['Pose_Class'].unique():
        pose_data = df[df['Pose_Class'] == pose_class]
        refined_data[pose_class] = {}
        
        for joint in joints:
            joint_angles = pose_data[joint].dropna()  # Drop missing values
            min_angle, max_angle = calculate_narrow_range(joint_angles, tolerance, min_range)
            refined_data[pose_class][f'{joint}_min'] = min_angle
            refined_data[pose_class][f'{joint}_max'] = max_angle
    
    return refined_data

# Define a function to save the refined angle data to CSV
def save_to_csv(refined_data, output_file):
    # Create a DataFrame from the refined data
    refined_df = pd.DataFrame.from_dict(refined_data, orient='index')
    refined_df.to_csv(output_file, index=True, index_label='Pose_Class')

# Example of usage
if __name__ == '__main__':
    # Load your angle dataset
    df = pd.read_csv('/Users/abcom/Documents/Yoga_Pose_Classification/csv_files/poses_angles1.csv')
    
    # List of joints for which angles are stored
    joints = ['armpit_left', 'elbow_left', 'armpit_right', 'elbow_right', 'hip_left', 
              'hip_right', 'knee_left', 'knee_right', 'ankle_left', 'ankle_right']
    
    # Process the angle data for each pose and joint, with tolerance and min range
    refined_angle_data = process_pose_angles_narrow(df, joints, tolerance=TOLERANCE_PERCENTAGE, min_range=MIN_RANGE)
    
    # Save the refined angle data to a new CSV
    save_to_csv(refined_angle_data, 'refined_pose_angles_with_tolerance.csv')

    print("Refined angle data with tolerance has been saved to 'refined_pose_angles_with_tolerance.csv'")
