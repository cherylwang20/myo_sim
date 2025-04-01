import mujoco
import numpy as np

model_path = r'C:\Users\chery\Documents\MyoLeg_Sarcopenia\mj_envs\robohive\simhive\myo_sim\leg\myolegs_abdomen.xml' 
model = mujoco.MjModel.from_xml_path(model_path)
sim = mujoco.MjData(model)
kf = model.keyframe('start')
sim.qpos = kf.qpos
mujoco.mj_forward(model, sim)

groups = {
    'torso': [('LSHO', 'RSHO')],
    'pelvis': [('RASI', 'LASI')],
    'thigh': [('RASI', 'RLFC'), ('LASI', 'LLFC')],
    'shank': [('RLFC', 'RLMAL'), ('LLFC', 'LLMAL')],
    'foot': [('RCAL', 'RTOE'), ('LCAL', 'LTOE')],
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

exp_distance = {'torso': {('LSHO', 'RSHO'): 0.30362862254684037}, 
                'pelvis': {('RASI', 'LASI'): 0.3221869814771535}, 
                'thigh': {('RASI', 'RLFC'): 0.4678452434164139, 
                          ('LASI', 'LLFC'): 0.4561921810837507}, 
                'shank': {('RLFC', 'RLMAL'): 0.3942182628596477, 
                          ('LLFC', 'LLMAL'): 0.4069699049818366},
                'foot': {('RCAL', 'RTOE'): 0.09640884513955036, 
                         ('LCAL', 'LTOE'): 0.09411012444298332}}

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