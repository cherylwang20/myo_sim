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
    'torso': [('LACR', 'RACR') ,('CLAV', 'C7'), ('C7', 'SACR'), ('CLAV', 'SACR'), ('RACR', 'SACR'), ('LACR', 'SACR')],
    'pelvis': [('RASI', 'LASI'), ('SACR', 'RASI'), ('SACR', 'LASI')],
    'thigh': [('RTRC', 'RLFE'), ('LTRC', 'LLFE'), ('RASI', 'RLFE'), ('LASI', 'LLFE')],
    'shank': [('RLFE', 'RLM'), ('LLFE', 'LLM')],
    'foot': [('RCAL', 'R1MT'), ('RCAL', 'R5MT'), ('LCAL', 'L1MT'), ('LCAL', 'L5MT')],
}

# Function to calculate component-wise distances
def component_distances(point1, point2):
    return np.abs(point1 - point2)

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
        distance = component_distances(site1_pos, site2_pos)
        distances[(site1, site2)] = distance
    distances_by_segment[segment] = distances

print(distances_by_segment)

# Load the .mat file
mat_data = loadmat('Subj04_jump.mat')
datastr = mat_data['Datastr'][0,0]
marker = datastr['Marker'][0,0]

# Extract marker labels and data
marker_labels = [str(label[0]) for label in marker['DataLabel'][0]]
marker_data = marker['MarkerData']
print(marker_data.shape)
marker_data = marker_data[:, [2, 0, 1], :]
marker_data = np.moveaxis(marker_data, 2, 0)

# Get first 10 timesteps and calculate average positions
first_10_timesteps = marker_data[:10]
average_positions = np.mean(first_10_timesteps, axis=0)

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
            distance = component_distances(site1_pos, site2_pos)
            distances[(site1, site2)] = distance
    exp_distance[segment] = distances

print(exp_distance)


# Calculate scaling ratios for each dimension (x, y, z)
scaling_ratios_by_segment = {}

for segment in exp_distance:
    if segment in distances_by_segment:
        segment_ratios = {'x': [], 'y': [], 'z': []}
        for pair in exp_distance[segment]:
            if pair in distances_by_segment[segment]:
                # Avoid division by zero
                sim_dist = distances_by_segment[segment][pair]
                exp_dist = exp_distance[segment][pair]
                
                for i, dim in enumerate(['x', 'y', 'z']):
                    if sim_dist[i] != 0:
                        ratio = exp_dist[i] / sim_dist[i]
                        segment_ratios[dim].append(ratio)
        
        # Calculate average ratios for each dimension
        avg_ratios = {}
        for dim in ['x', 'y', 'z']:
            if segment_ratios[dim]:
                avg_ratios[dim] = np.nanmean(segment_ratios[dim])
            else:
                avg_ratios[dim] = None
        
        scaling_ratios_by_segment[segment] = avg_ratios

# Print the results
print("Component-wise scaling ratios (x, y, z):")
for segment, ratios in scaling_ratios_by_segment.items():
    print(f"{segment}:")
    print(f"  X ratio: {ratios['x']:.4f}")
    print(f"  Y ratio: {ratios['y']:.4f}")
    print(f"  Z ratio: {ratios['z']:.4f}")
    print()