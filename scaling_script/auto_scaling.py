import xml.etree.ElementTree as ET
import os
import re
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
                new_pos = f'pos="{x:.6f} {y:.6f} {z:.6f}"'
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
    new_str="assets_test",
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
        new_str (str): New string to use (default: "assets_test")
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

from typing import Union, List, Tuple

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


if __name__ == "__main__":
    # creating a new main file for the scaled model
    old_xml_file = "./leg/myolegs_abdomen.xml"
    new_xml_file = "./leg/myolegs_abdomen_test.xml"

    replace_assets_path_and_increment_pos(old_xml_file, new_xml_file)


    # define the path to scale the leg mesh files
    leg_dir_path = "./leg/assets_test"
    torso_dir_path = "./torso/assets_test"

    check_directory(leg_dir_path)
    check_directory(torso_dir_path)


    #scaling the lower body part
    lower_input_xml = "./leg/assets/myolegs_assets.xml"
    lower_output_xml = "./leg/assets_test/myolegs_assets.xml"

    # Define body parts and their meshes + scaling
    leg_part_scales = {
        # Format: "body_part": (scale, ["mesh_pattern1", "mesh_pattern2", ...])
        "pelvis": (1.1879, ["r_pelvis", "l_pelvis"]),  # All meshes containing "pelvis" or "hip"
        "thigh": ([1.2148, 1.2148, 1.2148], ["r_femur", "l_femur"]),
        "shank": (1.0322, ["r_tibia","l_tibia", 'r_talus', 'l_talus']),
        "foot": (1.0330, ["r_foot", "l_foot", "r_bofoot", "l_bofoot"]),
    }

    scale_body_parts(lower_input_xml, lower_output_xml, leg_part_scales)

    #scaling torso part
    upper_input_xml = "./torso/assets/myotorso_abdomen_assets.xml"
    upper_output_xml = "./torso/assets_test/myotorso_abdomen_assets.xml"

    # Define body parts and their meshes + scaling
    upper_part_scales = {
        # Format: "body_part": (scale, ["mesh_pattern1", "mesh_pattern2", ...])
        "torso": (1.2657, ["sacrum", "hat_spine", "hat_jaw", "hat_skull", "hat_ribs_scap"]), 
    }

    scale_body_parts(upper_input_xml, upper_output_xml, upper_part_scales)

    # scaling the position of the upper body parts and sites
    torso_chain_input_xml = "./torso/assets/myotorso_abdomen_chain.xml"
    torso_chain_output_xml = "./torso/assets_test/myotorso_abdomen_chain.xml"
    
    # Define line ranges to scale
    upper_body_line_scales = [
        {
            'start_line': 14,
            'end_line': 58,
            'scale': 1.2657  # Uniform scaling
        }
    ]
    
    rescale_xml_by_line(torso_chain_input_xml, torso_chain_output_xml, upper_body_line_scales)

        # scaling the position of the upper body parts and sites
    leg_chain_input_xml = "./leg/assets/myolegs_chain.xml"
    leg_chain_output_xml = "./leg/assets_test/myolegs_chain.xml"
    
    # Define line ranges to scale
    leg_line_scales = [
        # pelvis
        {
            'start_line': 12,
            'end_line': 107,
            'scale': 1.2657  # Uniform scaling
        },
        # thigh
        {
            'start_line': 108,
            'end_line': 178,
            'scale': 1.2148  # Uniform scaling
        },
        {
            'start_line': 323,
            'end_line': 392,
            'scale': 1.2148  # Uniform scaling
        },
        {
            'start_line': 307,
            'end_line': 320,
            'scale': 1.2148  # Uniform scaling
        },
        {
            'start_line': 521,
            'end_line': 534,
            'scale': 1.2148  # Uniform scaling
        },
        # shank
        {
            'start_line': 179,
            'end_line': 245,
            'scale': 1.0322  # Uniform scaling
        },
        {
            'start_line': 393,
            'end_line': 459,
            'scale': 1.0322  # Uniform scaling
        },
        # foot
        {
            'start_line': 246,
            'end_line': 302,
            'scale': 1.033  # Uniform scaling
        },
        {
            'start_line': 460,
            'end_line': 516,
            'scale': 1.033  # Uniform scaling
        }
    ]
    
    rescale_xml_by_line(leg_chain_input_xml, leg_chain_output_xml, leg_line_scales)

    # defining the initial mass
    lower_mass_scales = {
        "pelvis": (1.1879, ["pelvis"]), 
        "thigh": (1.2148, ["femur_r", "femur_l"]),
        "shank": (1.0322, ["tibia_r","tibia_l"]),
        "foot": (1.0330, ["calcn_l", "calcn_r"]),
    }

    scale_body_part_masses(
        input_file="./leg/assets_test/myolegs_chain.xml",
        output_file="./leg/assets_test/myolegs_chain.xml",
        mass_scales=lower_mass_scales,
        target_total_mass=36.21,
        verbose=True
    )

    upper_mass_scales = {
        "torso": (1.2657, ["scarum", "torso"]), 
    }

    scale_body_part_masses(
        input_file="./torso/assets_test/myotorso_abdomen_chain.xml",
        output_file="./torso/assets_test/myotorso_abdomen_chain.xml",
        mass_scales=upper_mass_scales,
        target_total_mass=38.78,
        verbose=True
    )

    