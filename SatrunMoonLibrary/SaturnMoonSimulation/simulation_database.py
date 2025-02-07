import numpy as np

import os
import re
from datetime import datetime

import SaturnMoonLibrary.SaturnMoonSimulation as sms
import simulation

local_path_to_simulation_data =  os.path.join(".", "SaturnModelDatabase","simulation_data")
local_path_to_ring_data =  os.path.join(".", "SaturnModelDatabase","ring_data")

def create_header(
    filepath,
    data_dict,
    moon_count,
    initial_data_folder,
    epoch,
    dt,
    timesteps,
    num_test_particles,
    saved_points_modularity,
    skipped_timesteps,
    inner_radius,
    outer_radius,
    J2,
    ring_folder_name,
    number_of_radial_bins,
    number_of_azimuthal_bins,
    theta_max,
    include_shear_forces,
    include_particle_moon_collisions,
    integrator,
    initialisation_method,
    runtime
):
    """
    Creates a header for the binary files and prepends it to the existing file content.
    """
    moon_names = ", ".join(data_dict.keys())
    
    # Create a temporary file to write the header and existing content
    temp_filepath = filepath + '.tmp'
    
    with open(temp_filepath, 'wb') as temp_file:
        # Write the header information
        temp_file.write(f"Moon Names: {moon_names}\n".encode())
        temp_file.write(f"Moon Count: {moon_count}\n".encode())
        temp_file.write(f"Initial Data Folder: {initial_data_folder}\n".encode())
        temp_file.write(f"Epoch: {epoch}\n".encode())
        temp_file.write(f"dt: {dt}\n".encode())
        temp_file.write(f"Timesteps: {timesteps}\n".encode())
        temp_file.write(f"Number of Test Particles: {num_test_particles}\n".encode())
        temp_file.write(f"Saved Points Modularity: {saved_points_modularity}\n".encode())
        temp_file.write(f"Skipped Timesteps: {skipped_timesteps}\n".encode())
        temp_file.write(f"Inner Radius: {inner_radius}\n".encode())
        temp_file.write(f"Outer Radius: {outer_radius}\n".encode())
        temp_file.write(f"J2: {J2}\n".encode())
        temp_file.write(f"Ring Folder: {ring_folder_name}\n".encode())
        temp_file.write(f"Number of Radial Bins: {number_of_radial_bins}\n".encode())
        temp_file.write(f"Number of Azimuthal Bins: {number_of_azimuthal_bins}\n".encode())
        temp_file.write(f"Theta Max: {theta_max}\n".encode())
        temp_file.write(f"Include Shear Forces: {include_shear_forces}\n".encode())
        temp_file.write(f"Include Particle-Moon Collisions: {include_particle_moon_collisions}\n".encode())
        temp_file.write(f"Numerical Integrator: {integrator}\n".encode())
        temp_file.write(f"Initialisation Method: {initialisation_method}\n".encode())
        temp_file.write(f"Runtime: {runtime}\n".encode())
        # Write the end header marker
        temp_file.write(b"End of Header\n")

        # Append the content of the original file to the new file
        with open(filepath, 'rb') as original_file:
            temp_file.write(original_file.read())
    
    # Replace the original file with the new file
    import os
    os.replace(temp_filepath, filepath)

    print(f"Header created and prepended to {filepath}")


def read_header(filepath, output_type='string'):
    """
    Reads the header from the binary file until the 'End of Header' marker is found.
    By default, returns the header as a string. If output_type is 'dictionary', returns the header as a dictionary.
    """
    header = []

    # Open the binary file in read mode
    with open(filepath, 'rb') as file:
        while True:
            # Read line by line in binary mode
            line = file.readline()
            if b"End of Header" in line:
                break
            header.append(line)

    # Join the header lines into a single binary string and then decode
    header_string = b''.join(header).decode(errors='replace')  # Use 'replace' to handle undecodable bytes

    if output_type == 'dictionary':
        header_dict = {}
        # Split the header into lines
        lines = header_string.split('\n')
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                header_dict[key.strip()] = value.strip()
        return header_dict
    else:
        return header_string

def update_sublogfile(log_file_path, data_file_path, creation_date, source = "simulation"):
    """
    Update the log
    """
    # Creating data file_name
    creation_date_cleaned = creation_date.replace(":", "-").replace(" ", "_").replace(".", "-")

    if source == "simulation":
        data_filename = f"simulation {creation_date_cleaned}.bin"
    elif source == "horizons":
        data_filename = f"horizons_long {creation_date_cleaned}.bin"
    else:
        raise ValueError(f"Unknown source: {source}")

    # Creating Log string (Note: Header already includes a \n at the end)
    Extra_info_string = f"File Name: {data_filename} \n"+f"Creation Date: {creation_date}"
    header_string = read_header(data_file_path)
    seperation_string = "---------------------------------------------------------------------------------------------------------"
    log_entry = "\n" + seperation_string + "\n" + Extra_info_string + "\n" + header_string + seperation_string

    # Updating Log
    with open(log_file_path, 'a') as file:
        file.write(log_entry)

def parse_log_file(file_path):
    datasets = []
    with open(file_path, 'r') as file:
        content = file.read()

    # Skip content until the phrase "Below all datasets will be visible"
    start_index = content.find("Below all datasets will be visible:") + len('below all datasets will be visible:\n---------------------------------------------------------------------------------------------------------\n')
    if start_index != -1:
        content = content[start_index:]

    # Split content into sections for each dataset
    sections = content.split('---------------------------------------------------------------------------------------------------------\\n---------------------------------------------------------------------------------------------------------')
    if sections == [""]:
        return "Flag"
    

    # Extract data for each dataset
    for section in sections:
        labels = ['File Name','Moon Names', "Epoch", "dt","Timesteps","Number of Test Particles","Saved Points Modularity","Skipped Timesteps","Inner Radius","Outer Radius", "J2","Ring Folder","Number of Radial Bins","Number of Azimuthal Bins",'Theta Max',"Include Shear Forces","Include Particle-Moon Collisions","Numerical Integrator","Initialisation Method"]
        temp_dataset = {}
        for label in labels:
                groups = re.findall(f'{label}: (.*?)(?=\\n)', section)
                if groups == []:
                    temp_dataset[label] = "NULL"
                    continue
                else:
                    temp_dataset[label] = groups[0]

        datasets.append(temp_dataset)
    return datasets

def find_matching_simulation(
        file_path, 
        moon_names, 
        epoch, 
        dt,
        timesteps, 
        number_of_test_particles, 
        saved_points_modularity, 
        skipped_timesteps, 
        inner_radius, 
        outer_radius, 
        J2,
        ring_folder_name,
        number_of_radial_bins,
        number_of_azimuthal_bins,
        theta_max,
        include_shear_forces, 
        include_particle_moon_collisions,
        integrator,
        initialisation_method
    ):
    datasets = parse_log_file(file_path)
    # Convert moon name list to string
    moon_names = ["Saturn","Mimas", "Enceladus", "Tethys", "Dione", "Rhea", "Titan", "Hyperion", "Iapetus", "Phoebe", "Janus", "Epimetheus", "Helene", "Telesto", "Calypso", "Atlas", "Prometheus", "Pandora", "Pan", "Daphnis"]
    string = ", "
    moon_names = string.join(moon_names)

    if datasets == "Flag":
        return 0
    for dataset in datasets:
        print(dataset)
        if (dataset['Moon Names'] == moon_names and
                dataset['Epoch'] == str(epoch) and
                dataset['dt'] == str(dt) and
                dataset['Timesteps'] == str(timesteps) and
                dataset['Number of Test Particles'] == str(number_of_test_particles) and
                dataset['Saved Points Modularity'] == str(saved_points_modularity) and
                dataset['Skipped Timesteps'] == str(skipped_timesteps) and
                dataset['Inner Radius'] == str(inner_radius) and
                dataset['Outer Radius'] == str(outer_radius) and
                dataset['J2'] == str(J2) and
                dataset['Ring Folder'] == str(ring_folder_name) and
                dataset['Number of Radial Bins'] == str(number_of_radial_bins) and
                dataset['Number of Azimuthal Bins'] == str(number_of_azimuthal_bins) and
                dataset['Theta Max'] == str(theta_max) and
                dataset['Include Shear Forces'] == str(include_shear_forces) and
                dataset['Include Particle-Moon Collisions'] == str(include_particle_moon_collisions) and
                dataset['Numerical Integrator'] == str(integrator) and
                dataset['Initialisation Method'] == str(initialisation_method)
                ):
            return dataset['File Name']
    return 0

def extract_ring_info_out_path(local_path_ring):
    list_of_dir = os.listdir(local_path_ring)
    R_matrix_filename = [file for file in list_of_dir if "_r_" in file][0]
    z_matrix_filename = [file for file in list_of_dir if "_z_" in file][0]
    
    R_matrix_filepath = os.path.join(local_path_ring, R_matrix_filename)
    z_matrix_filepath = os.path.join(local_path_ring, z_matrix_filename)

    pattern = r"_(\d+)x(\d+)"

    match = re.search(pattern, R_matrix_filename)
    if match:
        matrix_rows = int(match.group(1))
        matrix_cols = int(match.group(2))
    else:
        raise Exception("Read Error: No match found w.r.t. matrix cols and rows")

    return R_matrix_filepath, z_matrix_filepath, matrix_rows, matrix_cols




def run_simulation(
    data_dict,
    epoch,
    dt,
    timesteps,
    num_test_particles,
    saved_points_modularity,
    skipped_timesteps,
    inner_radius,
    outer_radius,
    J2,
    matrix_foldername,
    number_of_radial_bins,
    number_of_azimuthal_bins,
    theta_max,
    include_shear_forces,
    include_particle_moon_collisions,
    integrator,
    initialisation_method,
    return_filepath = False
):
    """
    Runs the N-body simulation with the given parameters and generates a binary output file with simulation data.

    Parameters:
    - data_dict (dict): Dictionary containing data of moons, where keys are moon names and values are data of those moons.
    - epoch (list): The starting epoch for the simulation in .
    - dt (float): Time step for the simulation (in seconds).
    - timesteps (int): Total number of timesteps to simulate.
    - num_test_particles (int): Number of test particles in the simulation.
    - saved_points_modularity (int): Frequency of saving data points (e.g., save data every N steps).
    - skipped_timesteps (int): Number of initial timesteps to skip before saving any data.
    - inner_radius (float): Inner radius for initializing test particles (in meters).
    - outer_radius (float): Outer radius for initializing test particles (in meters).
    - J2 (float): J2 gravitational perturbation constant (default value for Saturn: -16290e-6).
    - matrix_foldername (str): Folder name from which to fetch ring potential data (R and z matrices).
    - number_of_radial_bins (int): Number of radial bins for collisions or shear (currently deprecated).
    - number_of_azimuthal_bins (int): Number of azimuthal bins for collision or shear (currently deprecated).
    - theta_max (float): The maximum angular extent for initializing test particles and preriodic boundaries if the Leapfrog Pizza slice integrator is choosen (in radians).
    - include_shear_forces (bool): Flag to include shear forces in the simulation (default is False, as currently deprecated).
    - include_particle_moon_collisions (bool): Flag to include particle-moon collisions (default is True).
    - integrator (str): Numerical integration method used in the simulation. Available options:
        - "Leapfrog": Standard leapfrog integrator, suitable for long-term stability.
        - "Leapfrog Pizza Slice": Leapfrog with periodic boundaries determined by theta_max innitialises automatically with Pizza Slice.
        - "Euler": Simple Euler method, not recommended for high accuracy as it's not symplectic.
        - "RK4": Fourth-order Runge-Kutta method, more accurate but slower and also not symplectic (possibly incorrect).
        - "Yoshida": 4th-order symplectic integrator for accurate simulations over long time scales.
        - "Yoshida Optimized": Optimized version of the Yoshida integrator.
        - "Yoshida 6th Order": Optimized verison of the 6th Order Yoshida.
        - "Yoshida 8th Order": Optimized verison of the 8th Order Yoshida.
    - initialisation_method (str): Method to initialize test particles. Available options:
        - "standard": Standard initialization of particles in a random distribution.
        - "linear": Linearly spaced initialization of particles.
        - "pizza slice": Initialization with particles arranged in a wedge or pizza-slice shape for specialized configurations.
    - return_filepath (bool): Whether to return the path to the generated output file. Default is False.

    Workflow:
    1. Gathers and rotates the data using `sms.get_horizons_data`.
    2. Creates a list of bodies for the simulation using `sms.generate_list_for_cpp_conversion`.
    3. Generates the output file name and path.
    4. Calculates the number of moons from the data dictionary.
    5. Runs the simulation using the `simulation.run_simulation` function.
    6. Creates a header for the binary output file using `create_header`.
    7. Updates the simulation log file with the simulation details.

    Returns:
    - None by default, unless `return_filepath` is set to True, in which case it returns the path to the output file.
    
    Raises:
    - Exception: If the input radii or other critical parameters are invalid.

    Example:
    >>> data_dict = {
            'Moon1': {'mass': 7.35e22, 'radius': 1.737e6, 'position': [0, 0, 0], 'velocity': [0, 0, 0]},
            'Moon2': {'mass': 6.42e23, 'radius': 3.4e6, 'position': [1e6, 0, 0], 'velocity': [0, 1e3, 0]}
        }
    >>> output = run_simulation(
            data_dict,
            epoch='2025-01-01T00:00:00',
            dt=1.0,
            timesteps=100000,
            num_test_particles=100,
            saved_points_modularity=10,
            skipped_timesteps=100,
            inner_radius=1e7,
            outer_radius=4e7,
            J2=-16290e-6,
            matrix_foldername='ring_data',
            number_of_radial_bins=200,
            number_of_azimuthal_bins=360,
            theta_max=2 * math.pi,
            include_shear_forces=False,
            include_particle_moon_collisions=True,
            integrator="Leapfrog",
            initialisation_method="standard",
            return_filepath=True
        )
    Simulation completed. Output file saved to: simulation_output.bin
    """
    # Error handeling w.r.t. the inputs
    if outer_radius <= inner_radius:
        raise Exception("Incorrect Radius Input: outer radius can't be smaler then inner radius of the rings")
    if saved_points_modularity == 0:
        raise Exception("Modulo Zero: Modulo zero is not defined")
    
    # Check if data already exists:
    moon_names = list(data_dict.keys())
    file_path = os.path.join(local_path_to_simulation_data,'simulation_log_file.txt')
    simulation_file = find_matching_simulation(
        file_path, 
        moon_names, 
        epoch, 
        dt,
        timesteps, 
        num_test_particles, 
        saved_points_modularity, 
        skipped_timesteps, 
        inner_radius, 
        outer_radius,
        J2, 
        matrix_foldername,
        number_of_radial_bins,
        number_of_azimuthal_bins,
        theta_max,
        include_shear_forces, 
        include_particle_moon_collisions,
        integrator,
        initialisation_method
    )
    print(simulation_file)
    if simulation_file != 0:
        print(f"Data already exists {simulation_file}")
        
    else:
        # Gathering and rotating data 
        rotated_dict, initial_data_folder = sms.get_horizons_data(data_dict,epoch,True,True)
        # initial_data_folder = "Temporary blank"

        # Creating the pybind11 list
        bodies_ls = sms.generate_list_for_cpp_conversion(rotated_dict)

        # Creating the output file name and path
        creation_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        creation_date_cleaned = creation_date.replace(":", "-").replace(" ", "_").replace(".", "-")
        output_filename = f"simulation {creation_date_cleaned}.bin"
        output_filepath = os.path.join(local_path_to_simulation_data, output_filename)

        # Calculating the Moon Count
        moon_count = len(data_dict.keys())

        # Potential Matrix related parameters
        ring_folder_path = os.path.join(local_path_to_ring_data, matrix_foldername)
        # Extracting with extract_ring_info_out_path
        R_matrix_filepath, z_matrix_filepath, matrix_rows, matrix_cols = extract_ring_info_out_path(ring_folder_path)

        # Debugging prints:
        # print(R_matrix_filepath, z_matrix_filepath,output_filepath)
        # print(matrix_rows,matrix_cols)

        # Running Simulation
        try:
            runtime = simulation.run_simulation(
            dt,
            timesteps,
            moon_count,
            num_test_particles,
            saved_points_modularity,
            skipped_timesteps,
            inner_radius,
            outer_radius,
            J2,
            bodies_ls,
            output_filepath,
            R_matrix_filepath,
            z_matrix_filepath,
            matrix_rows,
            matrix_cols,
            number_of_radial_bins,
            number_of_azimuthal_bins,
            theta_max,
            include_shear_forces,
            include_particle_moon_collisions,
            integrator,
            initialisation_method
            )
        except Exception as e:
            print(f"An error has occured in the void run simulation: {e}")
        
        # Create header
        create_header(
        output_filepath,
        data_dict,
        moon_count,
        initial_data_folder,
        epoch,
        dt,
        timesteps,
        num_test_particles,
        saved_points_modularity,
        skipped_timesteps,
        inner_radius,
        outer_radius,
        J2,
        matrix_foldername,
        number_of_radial_bins,
        number_of_azimuthal_bins,
        theta_max,
        include_shear_forces,
        include_particle_moon_collisions,
        integrator,
        initialisation_method,
        runtime
        )

        # Updating Log_file
        log_file_path = os.path.join(local_path_to_simulation_data,"simulation_log_file.txt")
        update_sublogfile(log_file_path, output_filepath, creation_date)

        if return_filepath:
            return output_filepath


def read_binary_file(filepath):
    """
    Reads a binary file containing simulation data and extracts the positional information of celestial bodies.

    The function reads the header of the binary file to obtain metadata about the simulation, such as the number of celestial bodies (moon count). It then reads the binary data, starting from the end of the header, and reshapes it into an array containing positional and velocity information for each body.

    Parameters:
    filepath (str): The path to the binary file containing the simulation data.

    Returns:
    np.ndarray: A 3D NumPy array where the first dimension corresponds to the timesteps, the second dimension corresponds to the number of bodies, and the third dimension contains six elements representing the position (x, y, z) and velocity (vx, vy, vz) of each body.
    """
    info_dict = read_header(filepath, output_type="dictionary")
    length_header = len(read_header(filepath))+ len("End of Header\n")
    num_bodies = int(info_dict['Moon Count'])+int(info_dict['Number of Test Particles'])
    data = np.fromfile(filepath, dtype=np.float64, offset = length_header)
    positions = data.reshape(-1, num_bodies, 6)
    return positions

