#import statements
import re

import pandas as pd

import simulation

def get_all_keys(saturn_data_dict):
    """
    Helper function to generate a list of all available keys in the subdictionaries.
    
    Parameters:
        saturn_data_dict (dict): Dictionary of moon names and their data.
        
    Returns:
        list: List of all keys found in the subdictionaries.
    """
    keys = set()
    for data in saturn_data_dict.values():
        keys.update(data.keys())
        # Include keys for r_i and v_i vectors
        for key in data.keys():
            if re.match(r"r_\d+", key) or re.match(r"v_\d+", key):
                keys.add(key + "_x")
                keys.add(key + "_y")
                keys.add(key + "_z")
    return list(keys)

def extract_time_tuples(saturn_data_dict):
    """
    Extract the t_i tuples from the saturn_data_dict, assuming all moons have the same time steps.
    
    Parameters:
        saturn_data_dict (dict): Dictionary of moon names and their data.
        
    Returns:
        dict: Dictionary with t_i tuples as keys and their values.
    """
    time_tuples = {}

    # Get the time steps from the first moon (assuming all moons have the same time steps)
    first_moon = next(iter(saturn_data_dict))
    for key, value in saturn_data_dict[first_moon].items():
        if key.startswith("t_"):
            time_tuples[key] = value
    
    return time_tuples

def convert_to_dataframe(saturn_data_dict, include_vectors=True, include_mass=True, include_time_independent=True, include_time=True):
    """
    Convert the saturn_data_dict to a pandas DataFrame and optionally unpack the r and v vectors, include mass, time-independent variables, and time tuples.
    
    Parameters:
        saturn_data_dict (dict): Dictionary of moon names and their data.
        include_vectors (bool): Whether to unpack the r and v vectors into separate columns. Default is True.
        include_mass (bool): Whether to include the Mass column. Default is True.
        include_time_independent (bool): Whether to include other time-independent variables. Default is True.
        include_time (bool): Whether to include the t_i tuples in the DataFrame. Default is True.
        
    Returns:
        DataFrame: DataFrame with moon names as objects and their data.
    """
    # Create a new dictionary to store the processed data
    processed_data = {}
    
    for moon, data in saturn_data_dict.items():
        # Copy data for modification
        processed_data[moon] = data.copy()
        
        if include_vectors:
            # Unpack r_0, r_i vectors
            for key in list(data.keys()):
                if re.match(r"r_\d+", key):
                    vector_key = key
                    processed_data[moon][f"{vector_key}_x"] = data[vector_key]["x"]
                    processed_data[moon][f"{vector_key}_y"] = data[vector_key]["y"]
                    processed_data[moon][f"{vector_key}_z"] = data[vector_key]["z"]
                    del processed_data[moon][vector_key]
            
            # Unpack v_0, v_i vectors
            for key in list(data.keys()):
                if re.match(r"v_\d+", key):
                    vector_key = key
                    processed_data[moon][f"{vector_key}_vx"] = data[vector_key]["vx"]
                    processed_data[moon][f"{vector_key}_vy"] = data[vector_key]["vy"]
                    processed_data[moon][f"{vector_key}_vz"] = data[vector_key]["vz"]
                    del processed_data[moon][vector_key]
        
        else:
            # Remove r_0, v_0 keys if not including vectors
            vector_keys = [key for key in data.keys() if re.match(r"(r|v)_\d+", key)]
            for vector_key in vector_keys:
                del processed_data[moon][vector_key]
        
        # Include time tuples if requested
        if include_time:
            for key in list(data.keys()):
                if re.match(r"t_\d+", key):
                    processed_data[moon][key] = data[key]
        else:
            # Remove time tuples if not including them
            time_keys = [key for key in data.keys() if re.match(r"t_\d+", key)]
            for time_key in time_keys:
                del processed_data[moon][time_key]

        # Remove mass if not included
        if not include_mass and "Mass" in processed_data[moon]:
            del processed_data[moon]["Mass"]
        
        # Remove time-independent variables if not included
        if not include_time_independent:
            time_independent_keys = ["semi-major axis", "eccentricity", "inclination"]  # Add more as needed
            for key in time_independent_keys:
                if key in processed_data[moon]:
                    del processed_data[moon][key]
    
    # Convert the dictionary to a DataFrame
    df = pd.DataFrame.from_dict(processed_data, orient='index')
    
    # Reset index to have a column for the moon names
    df.reset_index(inplace=True)
    
    # Rename columns
    df.rename(columns={'index': 'Object'}, inplace=True)
    
    return df

def convert_to_cpp_vector(saturn_data_dict):
    """
    Convert the saturn_data_dict to a string representing a C++ vector of Body objects.

    Parameters:
        saturn_data_dict (dict): Dictionary of moon names and their data.

    Returns:
        str: String representing a C++ vector of Body objects.
    """
    # Convert the dictionary to a DataFrame
    df = convert_to_dataframe(saturn_data_dict, include_vectors=True, include_mass=True, include_time_independent=False)
    
    # Initialize the C++ vector string
    cpp_vector_str = "std::vector<Body> bodies = { \n"
    
    # Iterate through the DataFrame and construct the C++ vector string
    for index, row in df.iterrows():
        position = [f"{row[f'r_{i}_x']}, {row[f'r_{i}_y']}, {row[f'r_{i}_z']}" for i in range(len([col for col in df.columns if re.match(r"r_\d+_x", col)]))]
        velocity = [f"{row[f'v_{i}_vx']}, {row[f'v_{i}_vy']}, {row[f'v_{i}_vz']}" for i in range(len([col for col in df.columns if re.match(r"v_\d+_vx", col)]))]

        radius = row["Radius"]  # Add radius from the DataFrame
        
        for pos, vel in zip(position, velocity):
            cpp_vector_str += f'    {{"{row["Object"]}", {row["Mass"]}, {radius}, {{{pos}}}, {{{vel}}}}},\n'
    
    # Close the vector string
    cpp_vector_str = cpp_vector_str.rstrip(",\n")  # Remove the trailing comma and newline
    cpp_vector_str += "\n};"
    
    return cpp_vector_str

def generate_list_for_cpp_conversion(data_dict):
    """
    Converts a dictionary of body data to a list of Body objects for C++ conversion.

    Parameters:
    data_dict (dict): A dictionary where keys are body names and values are dictionaries containing body data.

    Returns:
    list: A list of Body objects suitable for conversion to C++.

    Each entry in the returned list will have the following attributes:
    - name (str): The name of the body.
    - mass (float): The mass of the body.
    - radius (float): The radius of the body.
    - is_test_particle (bool): A flag indicating whether the body is a test particle (set to False).
    - pos (list of float): The position of the body, as a list of three floats [x, y, z].
    - vel (list of float): The velocity of the body, as a list of three floats [vx, vy, vz].
    - acc (list of float): The acceleration of the body, initialized to [0.0, 0.0, 0.0].
    
    Note:
    The function assumes that each value dictionary in data_dict contains the keys 'Mass', 'r_0', 'v_0', and 'Radius'.
    These keys are used to populate the corresponding attributes in the Body objects.
    """
    ls = []
    for key, body in data_dict.items():
        body = body.copy()
        body["name"] = key
        body["mass"] = body["Mass"]
        body["radius"] = body["Radius"]  # Add radius to the body
        body["is_test_particle"] = False
        body["pos"] = list(body["r_0"].values())
        body["vel"] = list(body["v_0"].values())
        body["acc"] = [0.0, 0.0, 0.0]

        # Keys to keep
        keys_to_keep = {"name", "mass", "radius", "is_test_particle", "pos", "vel", "acc"}

        # Delete keys not in keys_to_keep
        for k in list(body.keys()):  # list() is used to create a copy of keys to avoid modification during iteration
            if k not in keys_to_keep:
                del body[k]

        body_cpp = simulation.Body(body["name"], body["mass"], body["radius"], body["pos"], body["vel"], body["is_test_particle"])
        ls.append(body_cpp)

    return ls

def convert_to_txt_table(data_dict):
    """
    Convert the given data dictionary into a plain text table with horizontal lines between objects.

    Parameters:
        data_dict (dict): Dictionary of objects and their attributes.

    Returns:
        str: Formatted table as a string.
    """
    # Extract data for the table
    rows = []
    for obj, data in data_dict.items():
        # Extract necessary fields
        mass = data.get("Mass", "N/A")
        radius = data.get("Radius", "N/A")  # Include the radius
        r_0_x = data.get("r_0", {}).get("x", "N/A")
        r_0_y = data.get("r_0", {}).get("y", "N/A")
        r_0_z = data.get("r_0", {}).get("z", "N/A")
        v_0_vx = data.get("v_0", {}).get("vx", "N/A")
        v_0_vy = data.get("v_0", {}).get("vy", "N/A")
        v_0_vz = data.get("v_0", {}).get("vz", "N/A")

        rows.append([obj, mass, radius, r_0_x, r_0_y, r_0_z, v_0_vx, v_0_vy, v_0_vz])

    # Define column headers
    headers = [
        "Objects", "Mass (kg)", "Radius (m)", "r_0_x (m)", "r_0_y (m)", "r_0_z (m)",
        "v_0_vx (m/s)", "v_0_vy (m/s)", "v_0_vz (m/s)"
    ]

    # Calculate column widths
    column_widths = [max(len(str(row[i])) for row in rows + [headers]) for i in range(len(headers))]

    # Build table
    separator = "+-" + "-+-".join("-" * width for width in column_widths) + "-+"
    header_row = "| " + " | ".join(f"{headers[i]:<{column_widths[i]}}" for i in range(len(headers))) + " |"

    table = [separator, header_row, separator]

    for row in rows:
        table.append("| " + " | ".join(f"{str(row[i]):<{column_widths[i]}}" for i in range(len(row))) + " |")
        table.append(separator)  # Add separator after each row

    return "\n".join(table)
