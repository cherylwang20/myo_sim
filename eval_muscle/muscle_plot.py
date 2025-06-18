import mujoco
import numpy as np
import matplotlib.pyplot as plt

from xml.etree import ElementTree as ET

def fix_slashes_in_includes(xml_path, output_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for elem in root.iter("include"):
        if "file" in elem.attrib:
            elem.attrib["file"] = elem.attrib["file"].replace("\\", "/")
    tree.write(output_path)

# Example usage
fix_slashes_in_includes("./body/myofullbody_scaled.xml", "./body/myofullbody_scaled.xml")


# ---------- Configuration ----------
xml_paths = {
    "Default": "./body/myofullbody.xml",
    "Scaled": "./body/myofullbody_scaled.xml" 
}


muscle_name = "IL_L1_r" #"glmax1_r"
tendon_name = f"{muscle_name}_tendon"
joint_name = "flex_extension" #"hip_flexion_r"
joint_range = np.linspace(-1.35, 0.7, 50) #np.linspace(-0.55, 2, 50)
eps = 1e-4
# ------------------------------------

# Plot setup
plt.figure()

for label, xml_path in xml_paths.items():
    # Load model and data
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # Get indices
    tendon_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, tendon_name)
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    qpos_id = model.jnt_qposadr[joint_id]

    # Store moment arms
    moment_arms = []

    for q in joint_range:
        # q - eps
        data.qpos[:] = 0
        data.qpos[qpos_id] = q - eps
        mujoco.mj_forward(model, data)
        L1 = data.ten_length[tendon_id]

        # q + eps
        data.qpos[:] = 0
        data.qpos[qpos_id] = q + eps
        mujoco.mj_forward(model, data)
        L2 = data.ten_length[tendon_id]

        # Moment arm: -dL/dtheta
        r = -(L2 - L1) / (2 * eps)
        moment_arms.append(r)

    # Plot for this model
    plt.plot(joint_range, moment_arms, label=label)

# Finalize plot
plt.xlabel(f"{joint_name} angle (rad)")
plt.ylabel("Moment Arm (m)")
plt.title(f"Moment Arm vs Joint Angle for {muscle_name}")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
plt.close()


activation = 1.0  # Activation level for the muscle
plt.figure()

for label, xml_path in xml_paths.items():
    # Load model
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # Get IDs
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, muscle_name)
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    qpos_id = model.jnt_qposadr[joint_id]

    # Storage for this model
    mtu_lengths = []
    muscle_forces = []

    for q in joint_range:
        data.qpos[:] = 0
        data.qpos[qpos_id] = q
        data.act[:] = 0
        data.act[actuator_id] = activation

        mujoco.mj_forward(model, data)

        mtu_len = data.actuator_length[actuator_id]
        force = -data.actuator_force[actuator_id]

        mtu_lengths.append(mtu_len)
        muscle_forces.append(force)
    
    # Plot for this model
    plt.plot(mtu_lengths, muscle_forces, label=label, linewidth = 2, alpha = 0.5)

# Finalize plot
plt.xlabel("Muscle-Tendon Unit Length (m)")
plt.ylabel("Muscle Force (N)")
plt.title(f"Force–Length Curve: {muscle_name}")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()