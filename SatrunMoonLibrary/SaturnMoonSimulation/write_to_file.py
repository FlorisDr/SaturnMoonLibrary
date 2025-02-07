#import statements
from astropy.time import Time

from datetime import datetime
import os
import re
import shutil

import SaturnMoonLibrary.SaturnMoonSimulation as sms

#local path to horizons,
local_path_to_horizons =  os.path.join(".", "SaturnModelDatabase","horizons_data")

def check_existing_datasets(epoch, data_dict, return_folder_name = False):
    log_file_path = os.path.join(local_path_to_horizons, "Information_about_the_initial_datasets.txt")
    if not os.path.exists(log_file_path):
        print("log file does not exist, path used:",log_file_path)
        if return_folder_name:
            return False , ""
        else: 
            return False

    start_time = Time(epoch[0], format="jd").iso
    
    with open(log_file_path, 'r') as file:
        log_content = file.read()
        
        # Find all blocks of text separated by horizontal lines
        blocks = re.split(r'[-]{97}', log_content)
        
        #print(blocks)
        for block in blocks:
            # Ensure both start_time and moon_names appear in the same block
            if f"Start Date: {start_time}" in block and all(moon in block for moon in data_dict.keys()):
                match = re.search(r'Foldername: ([^\s]+)', block)
                folder_name = match.group(1)
                file_path = os.path.join(local_path_to_horizons,folder_name,"rotated_dictionary_"+folder_name+".txt")
                temp_dict = read_dictionary_from_file(file_path)
                if temp_dict.keys() == data_dict.keys():
                    if return_folder_name:
                        return True, folder_name
                    else:
                        return True
    if return_folder_name:
        return False, ""
    else:
        return False

def update_log_file(folder_name, start_time, epoch, data_dict):
    log_file_path = os.path.join(local_path_to_horizons, "Information_about_the_initial_datasets.txt")
    julian_start_date = epoch[0]
    
    log_entry = f"""---------------------------------------------------------------------------------------------------------
Foldername: {folder_name}                               Creation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Start Date: {start_time}                               Julian Start Date: {julian_start_date}

Input Data:
"""
    log_entry += sms.convert_to_txt_table(data_dict)
    log_entry += """\n---------------------------------------------------------------------------------------------------------"""
    
    with open(log_file_path, 'a') as file:
        file.write(log_entry)

def read_dictionary_from_file(file_path):
    """
    Reads a dictionary from a text file.

    Parameters:
        file_path (str): Path to the text file.

    Returns:
        dict: The dictionary read from the file.
    """
    with open(file_path, 'r') as file:
        # Skip the first two lines
        file.readline()
        file.readline()
        
        # Read the dictionary part of the file
        dictionary_str = file.read()
        
        # Convert the string representation of the dictionary to a dictionary object
        data_dict = eval(dictionary_str)
        
    return data_dict

def get_horizons_data(data_dict, epoch, return_dict=False, return_foldername = False):
    boolean, folder_name = check_existing_datasets(epoch, data_dict, True)
    if boolean:
        print(f"Dataset with the same Julian start date and bodies already exists. Foldername: {folder_name}")
        if return_foldername and return_dict:
            file_path = os.path.join(local_path_to_horizons, folder_name, "rotated_dictionary_" + folder_name + ".txt")
            if os.path.exists(file_path):
                return read_dictionary_from_file(file_path) , folder_name
            else:
                print(f"File not found: {file_path}")
                return None

        elif return_dict:
            file_path = os.path.join(local_path_to_horizons, folder_name, "rotated_dictionary_" + folder_name + ".txt")
            if os.path.exists(file_path):
                return read_dictionary_from_file(file_path)
            else:
                print(f"File not found: {file_path}")
                return None
        elif return_foldername:
            return folder_name
        else:
            return

    # Making an identifier string for filenames
    moon_count = len(data_dict.keys())
    start_time = Time(epoch[0], format="jd").iso
    creation_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    identifier_string = f"moon_count_{moon_count}_start_date_{start_time}_creation_date_{creation_date}"
    
    # Replace characters that are not allowed in filenames
    identifier_string = identifier_string.replace(":", "-").replace(" ", "_").replace(".", "-")
    
    folder_name = "initial_data_" + identifier_string
    folder_path = os.path.join(local_path_to_horizons, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    path_figure = os.path.join(folder_path, f"rotation_figure_before_and_after_{identifier_string}.png")
    radii_dict = sms.get_saturn_moons_radii(data_dict, epoch)
    horizons_dict = sms.get_saturn_moons_vectors(radii_dict, epoch)
    rotated_dict = sms.rotate_data(horizons_dict, epoch, show_plots=True, translate_to_center_of_mass_frame=True, path=path_figure)
    cpp_vector_str = sms.convert_to_cpp_vector(rotated_dict)

    # Prepare header for the text file
    moon_names = ", ".join(data_dict.keys())
    header = f"Moon Names: {moon_names}\nStart Time: {start_time}\n\n"
    
    # Save the text file with the header and the rotated_dict
    txt_file_path = os.path.join(folder_path, f"rotated_dictionary_initial_data_{identifier_string}.txt")
    with open(txt_file_path, 'w') as file:
        file.write(header + str(rotated_dict))  # Create the file with the header and data

    # Save the text file with the header and cpp_vector_str
    txt_file_path_cpp = os.path.join(folder_path, f"initial_data_for_cpp_{identifier_string}.txt")
    with open(txt_file_path_cpp, 'w') as file:
        file.write(header + cpp_vector_str)  # Create the file with the header and data

    # Update the log file
    update_log_file(folder_name, start_time, epoch, rotated_dict)

    if return_dict and return_foldername:
        return rotated_dict, folder_name

    elif return_dict:
        return rotated_dict
    
    elif return_foldername:
        return folder_name


def delete_dataset(folder_name):
    # Construct paths using os.path.join for cross-platform compatibility
    folder_path = os.path.join(".", "horizons_data", folder_name)
    
    print(f"Attempting to delete folder: {folder_path}")

    # Delete the folder if it exists
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"Deleted folder: {folder_path}")
    else:
        print(f"Folder not found: {folder_path}")

    # Log file path
    log_file_path = os.path.join(".", local_path_to_horizons, "Information_about_the_initial_datasets.txt")
    if os.path.exists(log_file_path):
        with open(log_file_path, 'r') as file:
            lines = file.readlines()

        # Variables to track the state
        updated_lines = []
        in_dataset_section = False
        skip_block = False
        found_block = False

        for i, line in enumerate(lines):
            # Identify start of "Below all datasets will be visible" section
            if "Below all datasets will be visible" in line:
                in_dataset_section = True
                updated_lines.append(line)
                continue

            # While inside the dataset section, check for horizontal lines
            if in_dataset_section and line.strip() == "---------------------------------------------------------------------------------------------------------":
                if skip_block:
                    skip_block = False
                    continue  # Skip the current horizontal line
                else:
                    updated_lines.append(line)
                    continue

            # Skip lines in a block that contains the specified foldername
            if in_dataset_section and folder_name in line:
                skip_block = True
                found_block = True
                # Start skipping the block from the horizontal line before the folder name
                updated_lines = updated_lines[:-1]  # Remove the preceding horizontal line
                continue  # Start skipping the block

            # Add lines that are not part of the block to remove
            if not skip_block:
                updated_lines.append(line)

        if not found_block:
            print(f"\nNo matching block found for folder name: {folder_name}")

        # Write back the updated content
        try:
            with open(log_file_path, 'w') as file:
                file.writelines(updated_lines)
            print("\nLog file updated successfully.")
        except Exception as e:
            print(f"\nError updating log file: {e}")
    else:
        print(f"Log file not found: {log_file_path}")