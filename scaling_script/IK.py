import scipy.io
import os
import numpy as np
import mujoco
from scipy.optimize import minimize
from tqdm import tqdm
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import skvideo.io

def read_markers(mat_file_path):
    """
    Reads a MATLAB .mat file containing marker data and returns a dictionary
    mapping each label to its corresponding time-series 3D position data.
    """
    try:
        mat_data = scipy.io.loadmat(mat_file_path)
        
        datastr = mat_data['Datastr'][0, 0]
        resample = datastr['Resample'][0, 0]
        data = resample['Marker']
        labels = datastr['Marker']['DataLabel'][0, 0].flatten()

        marker_dict = {}
        for i, label in enumerate(labels):
            clean_label = label[0] if isinstance(label, np.ndarray) else label
            marker_xyz = data[:, :, i].T
            marker_xzy = marker_xyz[[2, 0, 1], :]# x, z, y order
            marker_dict[clean_label] = marker_xzy.T
        
        for label, pos in marker_dict.items():
            num_nans = np.isnan(pos).sum()
            print(f"Marker '{label}': {num_nans} NaN values")
        return marker_dict

    except Exception as e:
        print(f"Error reading MAT file: {e}")
        raise

path = os.getcwd()

def load_mat(task):
    file = f'{task}.mat'
    df_markers = read_markers(path + '/' + file)


    for marker in ['SACR']:#['RS2', 'LS2', 'RS3', 'LS3', 'RT1', 'RT2', 'RT3', 'LT1', 'LT2', 'LT3']:
        if marker in df_markers:
            del df_markers[marker]
    
    return df_markers



def motion(model_name, task, body= 'leg', num_frames = 500):

    mj_model = mujoco.MjModel.from_xml_path(
                path+ f"/{body}/{model_name}.xml"
                )
    mj_data = mujoco.MjData(mj_model)

    df_markers = load_mat(task)

    site_name_to_id = {name: mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, name) for name in df_markers.keys()}

    camera = "front"
    options_ref = mujoco.MjvOption()
    options_ref.flags[:] = 0
    options_ref.geomgroup[1:] = 0

    renderer_ref = mujoco.Renderer(mj_model, height=1080, width=1240)
    renderer_ref.scene.flags[:] = 0

    #renderer_ref.update_scene(mj_data, camera=camera)
    error = []
    qpos_traj = []
    frames = []

    for frame in tqdm(range(num_frames), desc="Frames optimized"):
        target_pos = []
        site_ids = []
        for name in df_markers.keys():
            marker_pos = df_markers[name][frame]
            if not np.any(np.isnan(marker_pos)):
                target_pos.append(marker_pos)
                site_ids.append(site_name_to_id[name])
        target_pos = np.array(target_pos)

        # Only optimize every 10th frame
        if frame % 1 == 0:
            def ik_loss(qpos):
                mj_data.qpos[:] = qpos
                mujoco.mj_forward(mj_model, mj_data)
                sim_marker_pos = np.array([mj_data.site_xpos[site_id] for site_id in site_ids])
                return np.sum((sim_marker_pos - target_pos) ** 2)

            # Start from previous pose for better convergence
            if frame == 0:
                qpos_init = mj_data.qpos.copy()
            else:
                qpos_init = qpos_traj[-1]

            result = minimize(ik_loss, qpos_init, method='L-BFGS-B')
            best_qpos = result.x
            loss = result.fun
            if not np.isnan(loss):
                error.append(loss)
        else:
            # Use last optimized qpos
            best_qpos = qpos_traj[-1]
            loss = np.nan  # Or compute if needed

        # Update state and render as before
        mj_data.qpos[:] = best_qpos
        mujoco.mj_forward(mj_model, mj_data)
        renderer_ref.update_scene(mj_data, camera=camera)
        frame_in = renderer_ref.render()
        frames.append(frame_in)
        qpos_traj.append(best_qpos)
        if frame % 10 == 0:
            tqdm.write(f"Frame {frame+1}/{num_frames}: final loss = {loss:.3f}" if not np.isnan(loss) else f"Frame {frame+1}/{num_frames}: (no optimization)")
    
    output_name = f'videos/playback_{model_name}_{task}.mp4'
    skvideo.io.vwrite(output_name, np.asarray(frames[5:]), inputdict={"-r": '100'}, outputdict={"-pix_fmt": "yuv420p"})

    output_base = f'qpos_traj_{model_name}'
    np.save(output_base + '.npy', output_base)
    print('Average error loss is:', np.average(error))
    return error

model_def = 'myolegs_abdomen'
model_scal = 'myolegs_abdomen_test'
model_full = 'myoFullBody'

qpos_read = np.load('qpos_traj_myoFullBody.npy')
#print(qpos_read[0])

#serror_full = motion(model_full, 'full_body')
error_full_2 = motion('myofullbody', 'Subj04_run_63', 'body')


'''
error_def = motion(model_def)
error_scale = motion(model_scal)


plt.figure(figsize=(4, 2))
plt.plot(error_def, label = 'default')
plt.plot(error_scale, label = 'scaled')
plt.legend()
plt.show()
'''
