import c3d
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import os
import numpy as np
import mujoco
from scipy.optimize import minimize
from tqdm import tqdm
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import skvideo.io

def load_c3d_marker_data(filepath):
    """
    Load marker positions from a C3D file using the c3d Python library.
    Returns a dictionary: {marker_name: (n_frames, 3)}.
    """
    with open(filepath, 'rb') as handle:
        reader = c3d.Reader(handle)
        labels = [label.strip() for label in reader.point_labels]
        frames = list(reader.read_frames())

    n_markers = len(labels)
    n_frames = len(frames)
    data = np.zeros((n_frames, n_markers, 3))

    for i, (_, points, _) in enumerate(frames):
        data[i] = points[:, :3]/1000  # Exclude residuals
    data[:, :, 2] -= 0.02
    
    #data = data[:, :, [0, 2, 1]]
    marker_dict = {labels[i]: data[:, i, :] for i in range(n_markers)}
    return marker_dict

def rename_markers(marker_dict, rename_map):
    """
    Rename keys in marker_dict according to rename_map.
    Any marker not in rename_map is kept as-is.
    """
    return {
        rename_map[old_name]: marker_dict[old_name]
        for old_name in rename_map
        if old_name in marker_dict
    }

def print_qpos_as_xml(best_qpos, key_name="frame_0000"):
    qpos_str = ' '.join(f"{x:.8f}" for x in best_qpos)
    xml = f'<key name="{key_name}" qpos="{qpos_str}"/>'
    print(xml)

# === Main Execution ===
if __name__ == "__main__":
    c3d_file = "../140_08.c3d"  # Adjust path if needed
    marker_dict = load_c3d_marker_data(c3d_file)

    print("Loaded markers:", list(marker_dict.keys()))
    rename_map = {
        'Subject:C7': 'C7',
        'Subject:CLAV': 'CLAV',
        'Subject:LSHO': 'LACR',
        'Subject:RSHO': 'RACR',
        'Subject:LFWT': 'LASI',
        'Subject:RWFT': 'RASI',
        'Subject:LKNE': 'LLFE',
        #'Subject:LSHN': 'LS4',
        'Subject:LANK': 'LLM',
        'Subject:LHEL': 'LCAL',
        'Subject:LMT5': 'L5MT',
        'Subject:LMT1': 'L1MT',
        'Subject:RKNE': 'RLFE',
        #'Subject:RSHN': 'RS2',
        'Subject:RANK': 'RLM',
        'Subject:RHEL': 'RCAL',
        'Subject:RMT5': 'R5MT',
        'Subject:RMT1': 'R1MT',
    }

    # Apply renaming
    renamed_dict = rename_markers(marker_dict, rename_map)

    for name, traj in renamed_dict.items():
        print(f"{name}: {traj[0]}") 

    def load_mat(task):
        df_markers = renamed_dict


        for marker in ['SACR']:#['RS2', 'LS2', 'RS3', 'LS3', 'RT1', 'RT2', 'RT3', 'LT1', 'LT2', 'LT3']:
            if marker in df_markers:
                del df_markers[marker]
        
        return df_markers



    def motion(model_name, task, body= 'leg', num_frames = 500):
        path = os.getcwd()

        mj_model = mujoco.MjModel.from_xml_path(
                    path+ f"/../{body}/{model_name}.xml"
                    )
        mj_data = mujoco.MjData(mj_model)

        df_markers = load_mat(task)

        site_name_to_id = {name: mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, name) for name in df_markers.keys()}

        camera = "side"
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
            print_qpos_as_xml(best_qpos, key_name="init_pose")
            mujoco.mj_forward(mj_model, mj_data)
            renderer_ref.update_scene(mj_data, camera=camera)
            frame_in = renderer_ref.render()
            frames.append(frame_in)
            qpos_traj.append(best_qpos)
            if frame % 10 == 0:
                tqdm.write(f"Frame {frame+1}/{num_frames}: final loss = {loss:.3f}" if not np.isnan(loss) else f"Frame {frame+1}/{num_frames}: (no optimization)")
        
        output_name = f'../videos/playback_{model_name}_{task}.mp4'
        skvideo.io.vwrite(output_name, np.asarray(frames[5:]), inputdict={"-r": '100'}, outputdict={"-pix_fmt": "yuv420p"})

        output_base = f'qpos_traj_{model_name}'
        np.save(output_base + '.npy', output_base)
        print('Average error loss is:', np.average(error))
        return error

    model_def = 'myolegs_abdomen'
    model_scal = 'myolegs_abdomen_test'
    model_full = 'myoFullBody'

    #serror_full = motion(model_full, 'full_body')
    error_full_2 = motion('myofullbody', 'yoga_1', 'body')

