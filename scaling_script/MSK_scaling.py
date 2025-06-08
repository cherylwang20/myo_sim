import mujoco
import numpy as np
from scipy.io import loadmat

model_path = r'C:\Users\chery\Documents\myo_sim\leg\myolegs_abdomen.xml' 
model = mujoco.MjModel.from_xml_path(model_path)
sim = mujoco.MjData(model)
kf = model.keyframe('start')
sim.qpos = kf.qpos
mujoco.mj_forward(model, sim)

groups = {
    'torso': [('LACR', 'RACR') , ('C7', 'SACR'), ('CLAV', 'SACR'), ('RACR', 'SACR'), ('LACR', 'SACR')],#('CLAV', 'C7'),
    'pelvis': [('RASI', 'LASI'), ('SACR', 'RASI'), ('SACR', 'LASI')],
    'thigh': [('RTRC', 'RLFE'), ('LTRC', 'LLFE'), ('RASI', 'LLFE'), ('LASI', 'RLFE'), ('RASI', 'RLFE'), ('LASI', 'LLFE')],# ('RT3', 'RT1'), ('LT3', 'LT1'),('RT2', 'RT1'), ('LT2', 'LT1')],
    'shank': [('RLFE', 'RLM'), ('LLFE', 'LLM') ,('RS3', 'RS1'), ('LS3', 'LS1'),('RS2', 'RS1'), ('LS2', 'LS1')],
    'foot': [('RCAL', 'R1MT'), ('RCAL', 'R5MT'), ('LCAL', 'L1MT'), ('LCAL', 'L5MT')],
    # Add other segments and pairs as needed
}

# Function to calculate Euclidean distance
def euclidean_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

# Dictionary to store the results
distances_by_segment = {}

# Calculate distances for each specified pair in each group
for segment, pairs in groups.items():
    distances = {}
    for site1, site2 in pairs:
        site1_id = model.site(site1).id
        site2_id = model.site(site2).id
        site1_pos = sim.site_xpos[site1_id]
        site2_pos = sim.site_xpos[site2_id]
        distance = euclidean_distance(site1_pos, site2_pos)
        distances[(site1, site2)] = distance
    distances_by_segment[segment] = distances

# Print the results
print(distances_by_segment)


# Load the .mat file
mat_data = loadmat('Subj04_jump.mat')  # Replace 'your_file.mat' with your actual file path

# Access the nested structure correctly
datastr = mat_data['Datastr'][0,0]  # Main structure
marker = datastr['Marker'][0,0]     # Marker substructure

# Extract marker labels and data
marker_labels = [str(label[0]) for label in marker['DataLabel'][0]]  # Array of label strings
marker_data = marker['MarkerData']      # Actual position data

marker_data = np.moveaxis(marker_data, 2, 0)  # New shape: (4780, 33, 3)

# Get first 10 timesteps
first_10_timesteps = marker_data[:10]  # Shape: (10, 33, 3)

# Calculate average position per marker (across first 10 timesteps)
average_positions = np.mean(first_10_timesteps, axis=0)  # Shape: (33, 3)


exp_marker = {
    label: {'pos': pos} 
    for label, pos in zip(marker_labels, average_positions)
}

exp_distance = {}

# Calculate distances for each specified pair in each group
for segment, pairs in groups.items():
    distances = {}
    for site1, site2 in pairs:
        if site1 in exp_marker and site2 in exp_marker:
            site1_pos = exp_marker[site1]['pos']
            site2_pos = exp_marker[site2]['pos']
            distance = euclidean_distance(site1_pos, site2_pos)
            distances[(site1, site2)] = distance
    exp_distance[segment] = distances

print(exp_distance)

average_ratios_by_segment = {}

# Calculate ratios for each specified pair in each group and then average these ratios
for segment in exp_distance:
    ratios = []
    if segment in distances_by_segment:
        for pair in exp_distance[segment]:
            if pair in distances_by_segment[segment]:
                # Calculate the ratio of exp_distance to distances_by_segment for the same pair
                if distances_by_segment[segment][pair] != 0:  # Ensure no division by zero
                    ratio = exp_distance[segment][pair] / distances_by_segment[segment][pair]
                    ratios.append(ratio)
                else:
                    ratios.append(None)  # Append None if division by zero occurs

        # Calculating the average ratio for the segment if ratios are available
        if ratios:
            average_ratios_by_segment[segment] = np.nanmean(ratios)  # Use nanmean to ignore None values
        else:
            average_ratios_by_segment[segment] = None  # No valid ratios to average

# Print or return the results
print(average_ratios_by_segment)