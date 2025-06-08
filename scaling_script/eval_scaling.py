
import os
import time
from urllib.parse import urlparse
from tqdm import tqdm
import mujoco
import numpy as np
import pandas as pd
import requests
import scipy
import skvideo.io

def read_markers(mat_file_path):
    """
    Reads a MATLAB .mat file containing marker data and returns a dictionary
    mapping each label to its corresponding time-series 3D position data.
    """
    try:
        mat_data = scipy.io.loadmat(mat_file_path)
        
        datastr = mat_data['Datastr'][0, 0]
        markers = datastr['Marker'][0, 0]

        data = markers['MarkerData']  
        labels = markers['DataLabel'].flatten()  

        marker_dict = {}
        for i, label in enumerate(labels):
            clean_label = label[0] if isinstance(label, np.ndarray) else label
            marker_dict[clean_label] = data[i, :, :].T

        return marker_dict

    except Exception as e:
        print(f"Error reading MAT file: {e}")
        raise

def read_IK(mat_file_path):
    """
    Reads a MATLAB .mat file containing IK data and returns a labeled DataFrame.
    Handles different MATLAB structure formats.
    """
    try:
        mat_data = scipy.io.loadmat(mat_file_path)
        
        # Method 1: Direct access if not nested
        if 'IKAngData' in mat_data and 'IKAngDataLabel' in mat_data:
            data = mat_data['IKAngData']
            labels = mat_data['IKAngDataLabel']
        
        # Method 2: Nested structure access
        else:
            # Navigate through the nested structure
            datastr = mat_data['Datastr'][0,0]
            resample = datastr['Resample'][0,0]
            sych = resample['Sych'][0,0]
            
            data = sych['IKAngData']
            labels = sych['IKAngDataLabel']
            
            # Handle 2D label array
            if labels.ndim == 2:
                labels = labels[0]
            
        # Convert labels to Python strings
        column_labels = [str(label[0]) if isinstance(label, np.ndarray) else str(label) 
                        for label in labels.flatten()]
        
        return pd.DataFrame(data, columns=column_labels)
        
    except Exception as e:
        print(f"Error reading MAT file: {e}")
        raise


path = os.getcwd()
file = 'Subj04_jump.mat'
df = read_IK(path + '/' + file)
df_markers = read_markers(path + '/' + file)

mj_model = mujoco.MjModel.from_xml_path(
    path+ "/leg/myolegs_abdomen_test.xml"
)
mj_data = mujoco.MjData(mj_model)

joint_names = [mj_model.joint(jn).name for jn in range(mj_model.njnt)]
site_names = [
                mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_SITE, i) 
                for i in range(mj_model.nsite)
            ]
subc = [c for c in df.columns if c in joint_names]

camera = mujoco.MjvCamera()
camera.azimuth = 100
camera.distance = 3
camera.elevation = -25.0
camera.lookat = [0,0,.75]
options_ref = mujoco.MjvOption()
options_ref.flags[:] = 0
options_ref.geomgroup[1:] = 0
renderer_ref = mujoco.Renderer(mj_model)
renderer_ref.scene.flags[:] = 0
frames=[]


for t in tqdm(range(10), desc="Rendering frames"):
        for jn in subc:
            mjc_j_idx = mj_model.joint(joint_names.index(jn)).qposadr
            mj_data.qpos[mjc_j_idx] = np.deg2rad(df[jn].loc[t])
            if "knee_angle" in jn:  # knee joints have negative sign in myosuite
                mj_data.qpos[mjc_j_idx] *= -1

        mujoco.mj_forward(mj_model, mj_data)
        renderer_ref.update_scene(mj_data, camera=camera)#, scene_option=options_ref)
        #print(mj_data.qpos.copy())
        frame = renderer_ref.render()
        frames.append(frame)

os.makedirs('videos', exist_ok = True)
output_name = 'videos/playback_mot.mp4'
skvideo.io.vwrite(output_name, np.asarray(frames),outputdict={"-pix_fmt": "yuv420p"})

site_dict = {}
for i in range(mj_model.nsite):
    name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_SITE, i)
    pos = mj_data.site_xpos[i].copy()  # It's a numpy array of shape (3,)
    site_dict[name] = pos

diff_dict = {}

for key in site_dict.keys() & df_markers.keys():  # Only keys present in both dicts
    diff = site_dict[key][0] - df_markers[key][0]
    diff_dict[key] = diff

for name, pos in df_markers.items():
    pos_new = [pos[0][2], pos[0][0], pos[0][1]]
    pos_str = " ".join(str(x) for x in pos_new)
    print(f'<site name="{name}_n2" pos="{pos_str}" size="0.01"/>')