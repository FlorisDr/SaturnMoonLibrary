import numpy as np
import os
from datetime import datetime
import SaturnMoonLibrary.SaturnMoonSimulation as sms

# Local path for saving horizons long-format data
local_path_to_horizons_long_format_data = os.path.join(".", "SaturnModelDatabase", "horizons_long_format_data")

def write_binary_file_in_chunks(filename, positions, chunk_size=1000):
    """
    Writes a NumPy array to a binary file in chunks.

    Args:
        filename (str): The file path where data will be written.
        positions (np.ndarray): The array of positions and velocities to write.
        chunk_size (int): The size of each chunk to write to the file.
    """
    try:
        with open(filename, 'ab') as f:
            for i in range(0, positions.shape[0], chunk_size):
                chunk = positions[i:i+chunk_size].flatten()
                chunk.tofile(f)
        print(f"Binary data successfully written to {filename}")
    except IOError as e:
        print(f"Error writing to binary file: {e}")
        raise

def create_header_horizons_long(filepath, data_dict, moon_count, initial_data_folder, epoch, dt, timesteps, saved_points_modularity, integrator="Horizons"):
    """
    Creates and writes a header for the binary file with metadata information.

    Args:
        filepath (str): The path of the binary file where the header will be written.
        data_dict (dict): Dictionary containing moon names and data.
        moon_count (int): The number of moons.
        initial_data_folder (str): The folder containing the initial data.
        epoch (str): The initial epoch of the data.
        dt (float): The timestep value (time interval between data points).
        timesteps (int): The total number of timesteps in the simulation.
        saved_points_modularity (int): The modularity at which points are saved.
        integrator (str, optional): The name of the numerical integrator used. Defaults to "Horizons".
    """
    try:
        moon_names = ", ".join(data_dict.keys())

        with open(filepath, 'wb') as file:
            # Write the header information
            file.write(f"Moon Names: {moon_names}\n".encode())
            file.write(f"Moon Count: {moon_count}\n".encode())
            file.write(f"Initial Data Folder: {initial_data_folder}\n".encode())
            file.write(f"Epoch: {epoch}\n".encode())
            file.write(f"dt: {dt}\n".encode())
            file.write(f"Timesteps: {timesteps}\n".encode())
            file.write(f"Saved Points Modularity: {saved_points_modularity}\n".encode())
            file.write(f"Skipped Timesteps: {0}\n".encode())
            file.write(f"Numerical Integrator: {integrator}\n".encode())
            file.write(b"End of Header\n")
        
        print(f"Header successfully created and written to {filepath}")
    except IOError as e:
        print(f"Error writing header to file: {e}")
        raise

def create_long_format_horizons_data(data_dict, epoch, dt, number_of_timesteps, number_of_saved_points):
    """
    Queries Saturn moon data, creates a binary file, and logs the data creation.

    Args:
        data_dict (dict): Dictionary containing the moon data.
        epoch (str): Initial epoch of the simulation.
        dt (float): Timestep between each data point.
        number_of_timesteps (int): Total number of timesteps in the simulation.
        number_of_saved_points (int): Number of points saved in the output data.
    """
    try:
        # Querying data from the SaturnMoonSimulation library
        saturn_data_long = sms.get_saturn_moons_vectors(
            data_dict, epoch, dt, number_of_timesteps, number_of_saved_points,
            include_time=True, units=True
        )
        # Rotate data (possibly to a different reference frame)
        x, v = sms.rotate_data_long(saturn_data_long, epoch)

        # Reshape the data: concatenate position and velocity, then transpose
        positions = np.transpose((np.concatenate((x, v), axis=1)), axes=[0, 2, 1])

        # Create a unique output filename with the current timestamp
        creation_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        creation_date_cleaned = creation_date.replace(":", "-").replace(" ", "_")
        output_filename = f"horizons_long {creation_date_cleaned}.bin"
        output_filepath = os.path.join(local_path_to_horizons_long_format_data, output_filename)

        # Retrieve initial data folder name
        initial_data_folder = sms.get_horizons_data(data_dict, epoch, return_dict=False, return_foldername=True)

        # Calculate saved points modularity
        saved_points_modularity = number_of_timesteps // (number_of_saved_points - 1)

        # Create a header for the binary file
        create_header_horizons_long(
            output_filepath, data_dict, moon_count=len(data_dict),
            initial_data_folder=initial_data_folder, epoch=epoch,
            dt=dt, timesteps=number_of_timesteps, saved_points_modularity=saved_points_modularity
        )

        # Write the positions and velocities to the binary file
        write_binary_file_in_chunks(output_filepath, positions)

        # Update the log file with details about the created file
        log_file_path = os.path.join(local_path_to_horizons_long_format_data, "horizons_long_format_log_file.txt")
        sms.update_sublogfile(log_file_path, output_filepath, creation_date, source="horizons")

        print(f"Long format horizons data successfully created at {output_filepath}")
    except Exception as e:
        print(f"Error during data creation: {e}")
        raise
