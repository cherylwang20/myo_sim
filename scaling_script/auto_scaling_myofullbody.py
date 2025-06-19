import xml.etree.ElementTree as ET
import os
import re
import shutil
from pathlib import Path


def scale_body_parts(input_file, output_file, body_part_scales):
    """
    Scale groups of meshes corresponding to body parts.
    
    Args:
        input_file (str): Input XML file path
        output_file (str): Output XML file path
        body_part_scales (dict): Format:
            {
                "body_part_name": (scale_factor, ["mesh1", "mesh2", ...]),
                ...
            }
            (scale_factor can be a float or [x,y,z] list)
    """
    tree = ET.parse(input_file)
    root = tree.getroot()

    # Process each body part
    for body_part, (scale_factor, mesh_patterns) in body_part_scales.items():
        # Convert scale_factor to a string (e.g., "1.1 1.1 1.1")
        if isinstance(scale_factor, (int, float)):
            scale_str = f"{scale_factor} {scale_factor} {scale_factor}"
        else:
            scale_str = ' '.join(map(str, scale_factor))
        
        # Find all matching meshes for this body part
        for elem in root.iter('mesh'):
            mesh_file = elem.get('file', '')
            mesh_name = elem.get('name', '')
            
            # Check if this mesh matches any pattern for the body part
            for pattern in mesh_patterns:
                if (pattern.lower() in mesh_file.lower() or 
                    pattern.lower() in mesh_name.lower()):
                    elem.set('scale', scale_str)
                    print(f"Scaled {mesh_file or mesh_name} ({body_part}) to {scale_str}")
                    break  # Avoid duplicate scaling if multiple patterns match

    # Save modified XML
    tree.write(output_file, encoding='utf-8', xml_declaration=True)
    print(f"Saved scaled model to {output_file}")

def check_directory(dir_path):
    # Check if directory exists
    if not os.path.exists(dir_path):
        # Create the directory
        os.makedirs(dir_path)
        print(f"Directory '{dir_path}' created.")
    else:
        print(f"Directory '{dir_path}' already exists.")

def rescale_xml_by_line(input_file, output_file, line_scales):
    """
    Rescale positions in XML by multiplying values within specific line ranges.
    
    Args:
        input_file (str): Path to input XML file
        output_file (str): Path to save modified XML file
        line_scales (list): List of scaling rules:
            [
                {
                    'start_line': 100,  # First line to process (1-based)
                    'end_line': 200,    # Last line to process
                    'scale': 1.12       # Scaling factor (or [x_scale, y_scale, z_scale])
                },
                ...
            ]
    """
    # Read the entire file
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    modified = False
    
    # Process each scaling rule
    for rule in line_scales:
        start = rule['start_line'] - 1  # Convert to 0-based index
        end = rule['end_line']
        scale = rule['scale']
        
        if isinstance(scale, (int, float)):
            scale = [scale] * 3
        
        # Process each line in range
        for i in range(start, min(end, len(lines))):
            line = lines[i]
            
            # Find position attributes (format: position="x y z")
            matches = re.finditer(r'pos="([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)"', line)
            
            for match in matches:
                x, y, z = map(float, match.groups())
                x *= scale[0]
                y *= scale[1]
                z *= scale[2]
                
                # Replace with scaled values
                new_pos = f'pos="{x:.3f} {y:.3f} {z:.3f}"'
                line = line[:match.start()] + new_pos + line[match.end():]
                modified = True
                lines[i] = line
    
    if not modified:
        print("No position attributes found in specified line ranges.")
        return
    
    # Create output directory if needed
    output_dir = Path(output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write modified file
    with open(output_file, 'w') as f:
        f.writelines(lines)
    
    print(f"Rescaled positions. Saved to {output_file}")

def replace_assets_path_and_increment_pos(
    input_file, 
    output_file, 
    old_str="assets", 
    new_str="assets_scaled",
    body_name="root",
    pos_increment=0.2
):
    """
    Replace asset paths in XML include tags and increment the last element of the pos attribute
    for a specific body element.
    
    Args:
        input_file (str): Path to input XML file
        output_file (str): Path to save modified XML file
        old_str (str): String to replace in paths (default: "assets")
        new_str (str): New string to use (default: "assets_scaled")
        body_name (str): Name of the body whose pos should be modified (default: "root")
        pos_increment (float): Value to add to the last element of pos (default: 0.2)
    """
    # Parse the XML file
    tree = ET.parse(input_file)
    root = tree.getroot()
    
    modified_assets = False
    modified_pos = False
    
    # Find all include elements
    for include_elem in root.findall('.//include'):
        file_path = include_elem.get('file')
        if file_path and old_str in file_path:
            # Split the path into parts
            path_parts = list(Path(file_path).parts)
            
            # Replace only the folder names, not the filename
            for i in range(len(path_parts) - 1):  # Skip the last part (filename)
                if path_parts[i] == old_str:
                    path_parts[i] = new_str
                    modified_assets = True
            
            if modified_assets:
                new_path = str(Path(*path_parts))
                include_elem.set('file', new_path)
                print(f"Replaced asset path: {file_path} -> {new_path}")
    
    # Find the body element with the specified name
    for body_elem in root.findall(f'.//body[@name="{body_name}"]'):
        pos = body_elem.get('pos')
        if pos:
            # Split the pos string into components
            pos_parts = pos.split()
            if len(pos_parts) == 3:  # Ensure it's a 3D position
                try:
                    # Convert to floats, increment last element, and convert back to string
                    x, y, z = map(float, pos_parts)
                    new_z = z + pos_increment
                    new_pos = f"{x} {y} {new_z}"
                    body_elem.set('pos', new_pos)
                    print(f"Modified pos: {pos} -> {new_pos}")
                    modified_pos = True
                except ValueError:
                    print(f"Could not parse pos values for body '{body_name}'")
    
    if not modified_assets and not modified_pos:
        print("No modifications were made.")
        return
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the modified XML
    tree.write(output_file, encoding='utf-8', xml_declaration=True)
    print(f"Modified XML saved to {output_file}")

def scale_body_part_masses(
    input_file: str,
    output_file: str,
    mass_scales: dict,
    target_total_mass: float,
    verbose: bool = True
) -> None:
    # Parse XML
    tree = ET.parse(input_file)
    root = tree.getroot()

    # First pass: Extract current masses and validate bodies exist
    current_masses = {}
    body_info = {}  # Will store {'body_name': {'group': x, 'scale': y}}
    missing_bodies = []
    
    # Build body information mapping
    for group_name, (scale, body_names) in mass_scales.items():
        for body_name in body_names:
            body_info[body_name] = {
                'group': group_name,
                'scale': scale
            }
    
    # Extract masses from XML
    for body_name in body_info:
        body_elem = root.find(f'.//body[@name="{body_name}"]')
        if body_elem is None:
            missing_bodies.append(body_name)
            continue
            
        inertial_elem = body_elem.find('inertial')
        if inertial_elem is None or 'mass' not in inertial_elem.attrib:
            missing_bodies.append(body_name)
            continue
            
        current_masses[body_name] = float(inertial_elem.get('mass'))

    # Error handling
    if missing_bodies:
        raise ValueError(f"Missing bodies: {', '.join(missing_bodies)}")

    # Calculate scaled masses (before normalization)
    scaled_masses = {
        name: mass * body_info[name]['scale']
        for name, mass in current_masses.items()
    }
    total_scaled_mass = sum(scaled_masses.values())
    normalization_factor = target_total_mass / total_scaled_mass

    # Apply scaling
    mass_updates = {}
    group_changes = {}
    
    for body_name, info in body_info.items():
        body_elem = root.find(f'.//body[@name="{body_name}"]')
        inertial_elem = body_elem.find('inertial')
        
        old_mass = current_masses[body_name]
        new_mass = old_mass * info['scale'] * normalization_factor
        
        inertial_elem.set('mass', str(new_mass))
        
        # Record updates
        mass_updates[body_name] = (old_mass, new_mass)
        
        # Track group changes
        group_name = info['group']
        if group_name not in group_changes:
            group_changes[group_name] = {
                'old_total': 0,
                'new_total': 0,
                'scale': info['scale']
            }
        group_changes[group_name]['old_total'] += old_mass
        group_changes[group_name]['new_total'] += new_mass

    # Reporting
    if verbose:
        print("="*55)
        print(f"{'Group':<15} {'Old Mass':>10} {'New Mass':>10} {'Scale':>8} {'Change':>10}")
        print("-"*55)
        for group, data in group_changes.items():
            print(f"{group:<15} {data['old_total']:10.4f} {data['new_total']:10.4f} "
                  f"{data['scale']:8.4f} {data['new_total']-data['old_total']:10.4f}")

        print("\n" + "="*55)
        print(f"{'Target total mass:':<20} {target_total_mass:.4f}")
        print(f"{'Actual total mass:':<20} {sum(data['new_total'] for data in group_changes.values()):.4f}")
        print("="*55)

    # Save output
    output_dir = Path(output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    tree.write(output_file, encoding='utf-8', xml_declaration=True)
    if verbose:
        print(f"\nSaved to: {output_file}")

def parse_and_replace_assets(xml_path, output_path):
    """
    Parses an XML file, replaces '/assets/' with '/assets_scaled/' 
    in both text and attribute values, and writes the result to a new file.

    Args:
        xml_path (str): Path to the input XML file.
        output_path (str): Path to the output XML file.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Compile a regex pattern to match '/assets/' only
    pattern = re.compile(r'(?<=/)assets(?=/)')

    def replace_assets(element):
        # Replace in element text
        if element.text and '/assets/' in element.text:
            element.text = pattern.sub('assets_scaled', element.text)
        # Replace in element attributes
        for key, value in element.attrib.items():
            if '/assets/' in value:
                element.attrib[key] = pattern.sub('assets_scaled', value)
        # Recursively process child elements
        for child in element:
            replace_assets(child)

    replace_assets(root)
    tree.write(output_path)

#scaling for torso
def scale_mujoco_muscle_xml(xml_path, output_path, scale_factor):
    """
    Scales muscle actuator parameters in a MuJoCo XML file.
    
    Args:
        xml_path (str): Path to the input XML file.
        output_path (str): Path to save the updated XML.
        scale_factor (float): Linear model scaling factor (e.g., 1.2).
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    def scale_gain_bias(gain_or_bias_str):
        # Convert string to list of floats
        prm = list(map(float, gain_or_bias_str.strip().split()))

        if len(prm) < 9:
            raise ValueError("gainprm or biasprm must have at least 9 values")

        # Apply scaling
        prm[2] *= scale_factor ** 2   # force
        prm[3] *= scale_factor        # scale
        prm[6] *= scale_factor        # vmax

        return ' '.join(f"{x:.4f}" for x in prm)
    
    def scale_lengthrange(lengthrange_str):
        vals = list(map(float, lengthrange_str.strip().split()))
        if len(vals) != 2:
            raise ValueError("lengthrange must have exactly 2 values")
        vals = [x * scale_factor for x in vals]
        return ' '.join(f"{x:.6f}" for x in vals)

    for actuator in root.findall('.//general'):
        if 'gainprm' in actuator.attrib:
            old = actuator.attrib['gainprm']
            actuator.attrib['gainprm'] = scale_gain_bias(old)

        if 'biasprm' in actuator.attrib:
            old = actuator.attrib['biasprm']
            actuator.attrib['biasprm'] = scale_gain_bias(old)

        if 'lengthrange' in actuator.attrib:
            try:
                actuator.attrib['lengthrange'] = scale_lengthrange(actuator.attrib['lengthrange'])
            except ValueError:
                continue

    tree.write(output_path)
    print(f"Muscle Rescale Updated XML saved to: {output_path}")

def scale_mujoco_muscle_xml_by_name(xml_path, output_path, muscle_groups_scale_dict):
    """
    Scales muscle actuator parameters in a MuJoCo XML file based on muscle name substrings.

    Args:
        xml_path (str): Path to the input XML file.
        output_path (str): Path to save the updated XML.
        muscle_groups_scale_dict (dict): Keys are tuples/lists of name substrings,
                                         values are scale factors (e.g., {("bicep",): 1.1}).
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    def scale_gain_bias(gain_or_bias_str, scale_factor):
        prm = list(map(float, gain_or_bias_str.strip().split()))
        if len(prm) < 9:
            raise ValueError("gainprm or biasprm must have at least 9 values")
        prm[2] *= scale_factor ** 2   # force
        prm[3] *= scale_factor        # scale
        prm[6] *= scale_factor        # vmax
        return ' '.join(f"{x:.4f}" for x in prm)

    def scale_lengthrange(lengthrange_str, scale_factor):
        vals = list(map(float, lengthrange_str.strip().split()))
        if len(vals) != 2:
            raise ValueError("lengthrange must have exactly 2 values")
        vals = [x * scale_factor for x in vals]
        return ' '.join(f"{x:.6f}" for x in vals)

    for actuator in root.findall('.//general'):
        name = actuator.attrib.get("name", "").lower()
        matched_scale = None

        for substrings, scale in muscle_groups_scale_dict.items():
            if any(sub in name for sub in substrings):
                matched_scale = scale
                break

        if matched_scale is not None:
            if 'gainprm' in actuator.attrib:
                actuator.attrib['gainprm'] = scale_gain_bias(actuator.attrib['gainprm'], matched_scale)

            if 'biasprm' in actuator.attrib:
                actuator.attrib['biasprm'] = scale_gain_bias(actuator.attrib['biasprm'], matched_scale)

            if 'lengthrange' in actuator.attrib:
                try:
                    actuator.attrib['lengthrange'] = scale_lengthrange(actuator.attrib['lengthrange'], matched_scale)
                except ValueError:
                    continue

    tree.write(output_path)
    print(f"Muscle rescaled XML saved to: {output_path}")

def scale_tendon_springlength_by_muscle_groups(xml_input_path, xml_output_path, muscle_groups_scale_dict):
    """
    Scales springlengths of tendons in a MuJoCo XML based on groups of muscle name substrings.

    Args:
        xml_input_path (str): Path to input MuJoCo XML.
        xml_output_path (str): Path to save modified XML.
        muscle_groups_scale_dict (dict): Keys are tuples or lists of name substrings,
                                         values are scale factors.
                                         e.g., {("bicep", "brachialis"): 1.1}
    """
    springlength_re = re.compile(r'springlength="([\d.eE+-]+)"')
    name_re = re.compile(r'name="([^"]+)"')

    with open(xml_input_path, "r") as f_in, open(xml_output_path, "w") as f_out:
        for line in f_in:
            if 'springlength=' in line and 'name=' in line:
                name_match = name_re.search(line)
                spring_match = springlength_re.search(line)

                if name_match and spring_match:
                    name = name_match.group(1).lower()
                    original_val = float(spring_match.group(1))

                    for substrings, scale in muscle_groups_scale_dict.items():
                        if any(sub in name for sub in substrings):
                            scaled_val = original_val * scale
                            line = springlength_re.sub(f'springlength="{scaled_val:.6f}"', line)
                            break  # Apply only first matching group

            f_out.write(line)

if __name__ == "__main__":
    # creating a new main file for the scaled model
    old_xml_file = "./body/myofullbody.xml"
    new_xml_file = "./body/myofullbody_scaled.xml"

    replace_assets_path_and_increment_pos(old_xml_file, new_xml_file)


    # define the path to scale the leg mesh files
    leg_dir_path = "./leg/assets_scaled"
    torso_dir_path = "./torso/assets_scaled"

    check_directory(leg_dir_path)
    check_directory(torso_dir_path)


    #scaling the lower body part
    lower_input_xml = "./leg/assets/myolegs_assets.xml"
    lower_output_xml = "./leg/assets_scaled/myolegs_assets.xml"

    scale_dict = {
        "pelvis": 1.11,
        "thigh": 1.08,
        "shank": 1.09,
        "foot": 0.97,
        "torso": 1.02
    }

    # Define body parts and their meshes + scaling
    leg_part_scales = {
        # Format: "body_part": (scale, ["mesh_pattern1", "mesh_pattern2", ...])
        "pelvis": (scale_dict['pelvis'], ["r_pelvis", "l_pelvis"]),  # All meshes containing "pelvis" or "hip"
        "thigh": (scale_dict['thigh'], ["r_femur", "l_femur"]),
        "shank": (scale_dict['shank'], ["r_tibia","l_tibia", 'r_talus', 'l_talus']),
        "foot": (scale_dict['foot'], ["r_foot", "l_foot", "r_bofoot", "l_bofoot"]),
    }

    scale_body_parts(lower_input_xml, lower_output_xml, leg_part_scales)

    #muscle_file = 'leg/assets/myolegs_muscle.xml'
    ## scaling the muscle properties in torso
    muscle_scales = {
        ("glmax1_r", "glmax2_r",
         "glmax3_r", "glmed1_r", 
         "glmed2_r", "glmed3_r",
         "glmin1_r", "glmin2_r", 
         "glmin3_r", "iliacus_r",
         "piri_r", "psoas_r",
         "tfl_r", "glmax1_l", "glmax2_l",
         "glmax3_l", "glmed1_l", 
         "glmed2_l", "glmed3_l",
         "glmin1_l", "glmin2_l", 
         "glmin3_l", "iliacus_l",
         "piri_l", "psoas_l",
         "tfl_l"): scale_dict['pelvis'],
        ("addbrev_r", "addlong_r", "addmagDist_r",
         "addmagIsch_r", "addmagMid_r", "addmagProx_r",
         "bflh_r", "bfsh_r", "grac_r"
         "gaslat_r", "gasmed_r", "recfem_r", 
         "sart_r", "semimem_r", "semiten_r",
         "vasint_r", "vaslat_r", "vasmed_r", 
         "addbrev_l", "addlong_l", "addmagDist_l",
         "addmagIsch_l", "addmagMid_l", "addmagProx_l",
         "bflh_l", "bfsh_l", "grac_l"
         "gaslat_l", "gasmed_l", "recfem_l", 
         "sart_l", "semimem_l", "semiten_l",
         "vasint_l", "vaslat_l", "vasmed_l"): scale_dict['thigh'],
        ("edl_r", "ehl_r", "fdl_r",
         "fhl_r", "perbrev_r", "perlong_r",
         "soleus_r", "tibant_r", 
         "tibpost_r", "edl_l", "ehl_l", "fdl_l",
         "fhl_l", "perbrev_l", "perlong_l",
         "soleus_l", "tibant_l", 
         "tibpost_l"): scale_dict['shank']
    }


    scale_mujoco_muscle_xml_by_name("./leg/assets/myolegs_muscle.xml", "./leg/assets_scaled/myolegs_muscle.xml", muscle_scales)


    tendon_scales = {
        ("glmax1_r_tendon", "glmax2_r_tendon",
         "glmax3_r_tendon", "glmed1_r_tendon", 
         "glmed2_r_tendon", "glmed3_r_tendon",
         "glmin1_r_tendon", "glmin2_r_tendon", 
         "glmin3_r_tendon", "iliacus_r_tendon",
         "piri_r_tendon", "psoas_r_tendon",
         "tfl_r_tendon", "glmax1_l_tendon", "glmax2_l_tendon",
         "glmax3_l_tendon", "glmed1_l_tendon", 
         "glmed2_l_tendon", "glmed3_l_tendon",
         "glmin1_l_tendon", "glmin2_l_tendon", 
         "glmin3_l_tendon", "iliacus_l_tendon",
         "piri_l_tendon", "psoas_l_tendon",
         "tfl_l_tendon"): scale_dict['pelvis'],
        ("addbrev_r_tendon", "addlong_r_tendon", "addmagDist_r_tendon",
         "addmagIsch_r_tendon", "addmagMid_r_tendon", "addmagProx_r_tendon",
         "bflh_r_tendon", "bfsh_r_tendon", "grac_r_tendon"
         "gaslat_r_tendon", "gasmed_r_tendon", "recfem_r_tendon", 
         "sart_r_tendon", "semimem_r_tendon", "semiten_r_tendon",
         "vasint_r_tendon", "vaslat_r_tendon", "vasmed_r_tendon", 
         "addbrev_l_tendon", "addlong_l_tendon", "addmagDist_l_tendon",
         "addmagIsch_l_tendon", "addmagMid_l_tendon", "addmagProx_l_tendon",
         "bflh_l_tendon", "bfsh_l_tendon", "grac_l_tendon"
         "gaslat_l_tendon", "gasmed_l_tendon", "recfem_l_tendon", 
         "sart_l_tendon", "semimem_l_tendon", "semiten_l_tendon",
         "vasint_l_tendon", "vaslat_l_tendon", "vasmed_l_tendon"): scale_dict['thigh'],
        ("edl_r_tendon", "ehl_r_tendon", "fdl_r_tendon",
         "fhl_r_tendon", "perbrev_r_tendon", "perlong_r_tendon",
         "soleus_r_tendon", "tibant_r_tendon", 
         "tibpost_r_tendon", "edl_l_tendon", "ehl_l_tendon", "fdl_l_tendon",
         "fhl_l_tendon", "perbrev_l_tendon", "perlong_l_tendon",
         "soleus_l_tendon", "tibant_l_tendon", 
         "tibpost_l_tendon"): scale_dict['shank']
    }

    scale_tendon_springlength_by_muscle_groups("./leg/assets/myolegs_tendon.xml", "./leg/assets_scaled/myolegs_tendon.xml", tendon_scales)

    # Destination directory
    dst_dir = 'leg/assets_scaled/'

    # Copy the file to the destination directory
    #shutil.copy(muscle_file , dst_dir)
    #shutil.copy(tendon_file , dst_dir)

    #scaling torso part
    upper_input_xml = "./torso/assets/myotorso_assets.xml"
    upper_output_xml = "./torso/assets_scaled/myotorso_assets.xml"

    # Define body parts and their meshes + scaling
    upper_part_scales = {
        # Format: "body_part": (scale, ["mesh_pattern1", "mesh_pattern2", ...])
        "torso": (scale_dict['torso'], ["sacrum", "lumbar5", "lumbar4", "lumbar3", "lumbar2", "lumbar1", 
                                        "torso_geom_1", "torso_geom_2", "torso_geom_3", "torso_geom_4", 
                                        "torso_geom_5", "torso_geom_6", "torso_geom_7", "torso_geom_8",
                                        "torso_geom_9", "torso_geom_10", "torso_geom_11", "torso_geom_12", "torso_geom_13"]), 
    }

    scale_body_parts(upper_input_xml, upper_output_xml, upper_part_scales)

    ## scaling the muscle properties in torso
    scale_mujoco_muscle_xml(
        xml_path=upper_output_xml,
        output_path=upper_output_xml,
        scale_factor=scale_dict['torso']
    )

    #scaling head
    head_input_xml = "./head/assets/myohead_simple_assets.xml"
    head_output_xml = "./head/assets_scaled/myohead_simple_assets.xml"

    # Define body parts and their meshes + scaling
    head_part_scales = {
        # Format: "body_part": (scale, ["mesh_pattern1", "mesh_pattern2", ...])
        "head": (scale_dict['torso'], ["hat_cervical", "hat_jaw", "hat_skull"]), 
    }

    scale_body_parts(head_input_xml, head_output_xml, head_part_scales)

    # scaling the position of the upper body parts and sites
    torso_chain_input_xml = "./torso/assets/myotorso_chain.xml"
    torso_chain_output_xml = "./torso/assets_scaled/myotorso_chain.xml"
    
    # Define line ranges to scale
    upper_body_line_scales = [
        {
            'start_line': 9,
            'end_line': 702,
            'scale': scale_dict['torso']  # Uniform scaling
        }
    ]
    
    rescale_xml_by_line(torso_chain_input_xml, torso_chain_output_xml, upper_body_line_scales)

    #scaling the head, since now it is defined separately
     # scaling the position of the upper body parts and sites
    head_chain_input_xml = "./head/assets/myohead_rigid_chain.xml"
    head_chain_output_xml = "./head/assets_scaled/myohead_rigid_chain.xml"
    
    # Define line ranges to scale
    head_line_scales = [
        {
            'start_line': 10,
            'end_line': 23,
            'scale': scale_dict['torso']  # Uniform scaling
        }
    ]
    
    rescale_xml_by_line(head_chain_input_xml, head_chain_output_xml, head_line_scales)

       # scaling the position of the upper body parts and sites
    leg_chain_input_xml = "./leg/assets/myolegs_chain.xml"
    leg_chain_output_xml = "./leg/assets_scaled/myolegs_chain.xml"
    
    # Define line ranges to scale
    leg_line_scales = [
        # pelvis
        {
            'start_line': 12,
            'end_line': 107,
            'scale': scale_dict['pelvis']  # Uniform scaling
        },
        # thigh
        {
            'start_line': 108,
            'end_line': 178,
            'scale': scale_dict['thigh']  # Uniform scaling
        },
        {
            'start_line': 323,
            'end_line': 392,
            'scale': scale_dict['thigh']  # Uniform scaling
        },
        {
            'start_line': 307,
            'end_line': 320,
            'scale': scale_dict['thigh']  # Uniform scaling
        },
        {
            'start_line': 521,
            'end_line': 534,
            'scale': scale_dict['thigh']  # Uniform scaling
        },
        # shank
        {
            'start_line': 179,
            'end_line': 253,
            'scale': scale_dict['shank']  # Uniform scaling
        },
        {
            'start_line': 393,
            'end_line': 465,
            'scale': scale_dict['shank']  # Uniform scaling
        },
        # foot
        {
            'start_line': 287,
            'end_line': 302,
            'scale': scale_dict['foot']  # Uniform scaling
        },
        {
            'start_line': 502,
            'end_line': 516,
            'scale': scale_dict['foot']  # Uniform scaling
        }
    ]
    
    rescale_xml_by_line(leg_chain_input_xml, leg_chain_output_xml, leg_line_scales)

    # defining the initial mass
    lower_mass_scales = {
        "pelvis": (scale_dict['pelvis'], ["pelvis"]), 
        "thigh": (scale_dict['thigh'], ["femur_r", "femur_l"]),
        "shank": (scale_dict['shank'], ["tibia_r","tibia_l"]),
        "foot": (scale_dict['foot'], ["calcn_l", "calcn_r"]),
    }

    scale_body_part_masses(
        input_file="./leg/assets_scaled/myolegs_chain.xml",
        output_file="./leg/assets_scaled/myolegs_chain.xml",
        mass_scales=lower_mass_scales,
        target_total_mass=36.21,
        verbose=True
    )

    upper_mass_scales = {
        "torso": (scale_dict['torso'], ["sacrum", "lumbar5", "lumbar4", "lumbar3", "lumbar2", "lumbar1", "torso"]), 
    }

    scale_body_part_masses(
        input_file="./torso/assets_scaled/myotorso_chain.xml",
        output_file="./torso/assets_scaled/myotorso_chain.xml",
        mass_scales=upper_mass_scales,
        target_total_mass=38.78,
        verbose=True
    )

    #replace the asset path for head in torso with scaled one:
    parse_and_replace_assets(r'.\torso\assets_scaled\myotorso_chain.xml', r'.\torso\assets_scaled\myotorso_chain.xml')
    parse_and_replace_assets(r'.\torso\assets_scaled\myotorso_assets.xml', r'.\torso\assets_scaled\myotorso_assets.xml') 




    