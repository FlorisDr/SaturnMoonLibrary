# Import statements
import numpy as np
import matplotlib.pyplot as plt

import SaturnMoonLibrary.SaturnMoonSimulation as sms

# Function to rotate a vector to align with the z-axis
def rot_to_top(h):
    """
    Rotates the given vector `h` to align with the z-axis.
    
    Parameters:
        h (numpy.ndarray): 3D vector to be rotated.

    Returns:
        numpy.ndarray: Rotation matrix to align `h` with the z-axis.

    Raises:
        ValueError: If the input vector has incorrect dimensions.
    """
    if len(h) != 3:
        raise ValueError("Input vector `h` must have exactly 3 elements.")
    
    x, y, z = h / np.sqrt(np.dot(h, h))
    s = np.sqrt(x**2 + y**2)
    r = np.sqrt(x**2 + y**2 + z**2)

    # Special case: no rotation needed if vector is already along the z-axis
    if x == y == 0:
        return np.eye(3)

    rotation_matrix = np.matmul(
        np.array([[z / r, 0, -s / r], [0, 1, 0], [s / r, 0, z / r]]),
        np.array([[x / s, y / s, 0], [-y / s, x / s, 0], [0, 0, 1]])
    )

    return rotation_matrix

# Function to create position, velocity, and mass arrays from the data dictionary
def create_arrays(data_dict):
    """
    Creates position, velocity, and mass arrays from a dictionary of moon data.
    
    Parameters:
        data_dict (dict): Dictionary containing data of moons and Saturn.
        
    Returns:
        (numpy.ndarray, numpy.ndarray, numpy.ndarray): Arrays of position (x), velocity (v), and mass (m).
        
    Raises:
        KeyError: If required keys are missing in the input dictionary.
    """
    num_moons = len(data_dict)
    x = np.zeros((3, num_moons))
    v = np.zeros((3, num_moons))
    m = np.zeros(num_moons)

    for i, (key, item) in enumerate(data_dict.items()):
        try:
            for k, coord in enumerate(["x", "y", "z"]):
                x[k, i] = item["r_0"][coord] - data_dict["Saturn"]["r_0"][coord]
                v[k, i] = item["v_0"]["v" + coord] - data_dict["Saturn"]["v_0"]["v" + coord]
            m[i] = item["Mass"]
        except KeyError as e:
            raise KeyError(f"Missing key in data dictionary: {e}")

    return x, v, m

def create_arrays_long(data_dict):
    """
    Creates position, velocity, and mass arrays from a dictionary with multiple timesteps.
    
    Parameters:
        data_dict (dict): Dictionary containing data of moons and Saturn for multiple timesteps.
        
    Returns:
        (numpy.ndarray, numpy.ndarray, numpy.ndarray): Arrays of position (x), velocity (v), and mass (m).
        
    Raises:
        KeyError: If required keys are missing in the input dictionary.
    """
    timesteps = len(sms.extract_time_tuples(data_dict))
    num_moons = len(data_dict)
    
    x = np.zeros((timesteps, 3, num_moons))
    v = np.zeros((timesteps, 3, num_moons))
    m = np.zeros(num_moons)

    for i, (key, item) in enumerate(data_dict.items()):
        try:
            for k, coord in enumerate(["x", "y", "z"]):
                for t in range(timesteps):
                    x[t, k, i] = item[f"r_{t}"][coord] - data_dict["Saturn"][f"r_{t}"][coord]
                    v[t, k, i] = item[f"v_{t}"]["v" + coord] - data_dict["Saturn"][f"v_{t}"]["v" + coord]
            m[i] = item["Mass"]
        except KeyError as e:
            raise KeyError(f"Missing key in data dictionary: {e}")

    return x, v, m

# Function to rotate position and velocity arrays
def rotate_arrays(x, v, epoch):
    """
    Rotates position and velocity arrays to align with the angular momentum of Pan.
    
    Parameters:
        x (numpy.ndarray): Position array.
        v (numpy.ndarray): Velocity array.
        epoch (list): List of Julian dates.

    Returns:
        (numpy.ndarray, numpy.ndarray): Rotated position and velocity arrays.
    """
    # Pan's angular momentum vector aligns the x-y plane with Saturn's equator
    pan_dict = {"Saturn": {"ID": 699, "Mass": 5.6834e+26}, "Pan": {"ID": 618, "Mass": 4.95e+15}}
    saturn_data_pan = sms.get_saturn_moons_vectors(pan_dict, epoch)
    
    # Get Pan's position, velocity, and mass
    xp, vp, mp = create_arrays(saturn_data_pan)

    # Calculate angular momentum of Pan
    angular_momentum_pan = np.cross(xp, mp * vp, axis=0)[:, -1]

    # Rotate arrays
    rotation_matrix = rot_to_top(angular_momentum_pan)
    rotated_x = rotation_matrix @ x
    rotated_v = rotation_matrix @ v

    return rotated_x, rotated_v

# Function to transform arrays back into the dictionary format
def arrays_to_dict(x, v, m, data_dict):
    """
    Transforms position and velocity arrays back into dictionary format.
    
    Parameters:
        x (numpy.ndarray): Position array (2D or 3D).
        v (numpy.ndarray): Velocity array (2D or 3D).
        m (numpy.ndarray): Mass array.
        data_dict (dict): Original dictionary with moon and Saturn data.
    
    Returns:
        dict: Transformed dictionary with updated position and velocity data.
    
    Raises:
        TypeError: If input arrays `x` or `v` are not of type numpy.ndarray.
        ValueError: If input arrays do not have 2 or 3 dimensions.
    """
    transformed_dict = {}
    keys = list(data_dict.keys())

    for i, key in enumerate(keys):
        transformed_dict[key] = {}
        for k, value in data_dict[key].items():
            transformed_dict[key][k] = value

        transformed_dict[key]["Mass"] = m[i]

        if isinstance(x, np.ndarray):
            if x.ndim == 2:
                transformed_dict[key]["r_0"] = {
                    "x": x[0, i],
                    "y": x[1, i],
                    "z": x[2, i]
                }
            elif x.ndim == 3:
                for t in range(len(x)):
                    transformed_dict[key][f"r_{t}"] = {
                        "x": x[t, 0, i],
                        "y": x[t, 1, i],
                        "z": x[t, 2, i]
                    }
            else:
                raise ValueError(f"x should have 2 or 3 dimensions, currently has {x.ndim}.")
        else:
            raise TypeError("x should be a numpy array.")

        if isinstance(v, np.ndarray):
            if v.ndim == 2:
                transformed_dict[key]["v_0"] = {
                    "vx": v[0, i],
                    "vy": v[1, i],
                    "vz": v[2, i]
                }
            elif v.ndim == 3:
                for t in range(len(v)):
                    transformed_dict[key][f"v_{t}"] = {
                        "vx": v[t, 0, i],
                        "vy": v[t, 1, i],
                        "vz": v[t, 2, i]
                    }
            else:
                raise ValueError(f"v should have 2 or 3 dimensions, currently has {v.ndim}.")
        else:
            raise TypeError("v should be a numpy array.")
    
    return transformed_dict

def calculate_center_of_mass(x, m):
    """
    Calculate the center of mass position.
    
    Parameters:
        x (numpy.ndarray): Array of shape (3, N) containing position vectors.
        m (numpy.ndarray): Array of shape (N,) containing masses.
        
    Returns:
        numpy.ndarray: Center of mass position.
        
    Raises:
        ValueError: If the dimensions of `x` and `m` do not match.
    """
    if x.shape[1] != len(m):
        raise ValueError("Mismatch between the number of objects in `x` and `m`.")
    
    total_mass = np.sum(m)
    if total_mass == 0:
        raise ValueError("Total mass cannot be zero.")
    
    center_of_mass_pos = np.sum(x * m, axis=1) / total_mass
    return center_of_mass_pos


def translate_to_center_of_mass(x, v, m):
    """
    Translate the positions and velocities so that the center of mass of the system is at the origin (0,0,0).
    
    Parameters:
        x (numpy.ndarray): Array of shape (3, N) containing position vectors.
        v (numpy.ndarray): Array of shape (3, N) containing velocity vectors.
        m (numpy.ndarray): Array of shape (N,) containing masses.
        
    Returns:
        (numpy.ndarray, numpy.ndarray): Translated position and velocity arrays.
        
    Raises:
        ValueError: If the dimensions of `x`, `v`, or `m` are not compatible.
    """
    if x.shape != v.shape or x.shape[1] != len(m):
        raise ValueError("Dimensions of `x`, `v`, and `m` must match.")
    
    # Calculate the center of mass position
    center_of_mass_pos = calculate_center_of_mass(x, m)
    
    # Translate positions
    translated_x = x - center_of_mass_pos[:, np.newaxis]

    # Total mass
    total_mass = np.sum(m)
    if total_mass == 0:
        raise ValueError("Total mass cannot be zero.")
    
    # Calculate the center of mass velocity
    center_of_mass_vel = np.sum(v * m, axis=1) / total_mass
    
    # Translate velocities
    translated_v = v - center_of_mass_vel[:, np.newaxis]
    
    return translated_x, translated_v


def plot_rotated_data(x, v, m, rotated_x, rotated_v, saturn_index=0, path=None):
    """
    Plot the data before and after rotation in 3D.

    Parameters:
        x (numpy.ndarray): Original position array.
        v (numpy.ndarray): Original velocity array.
        m (numpy.ndarray): Mass array.
        rotated_x (numpy.ndarray): Rotated position array.
        rotated_v (numpy.ndarray): Rotated velocity array.
        saturn_index (int): Index of Saturn in the data arrays.
        path (str): Path to save the plot. If None, the plot will not be saved.
    
    Raises:
        ValueError: If the input arrays do not have compatible dimensions.
    """
    if x.shape != v.shape or rotated_x.shape != rotated_v.shape:
        raise ValueError("Mismatched dimensions between position and velocity arrays.")

    fig = plt.figure(figsize=(18, 8))

    # Calculate center of mass for original data
    center_of_mass = calculate_center_of_mass(x, m)

    # Original data subplot
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_xlim((-2 * 10**8, 2 * 10**8))
    ax1.set_ylim((-2 * 10**8, 2 * 10**8))
    ax1.set_zlim((-2 * 10**8, 2 * 10**8))
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_box_aspect(None, zoom=0.92)
    ax1.set_title("Before Correction")
    
    # Axis labels
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_zlabel('z (m)')
    
    # Plot original positions
    ax1.plot(x[0], x[1], x[2], ".", label='Moons')
    
    # Highlight Saturn in gold
    ax1.plot(x[0, saturn_index], x[1, saturn_index], x[2, saturn_index], "o", color="gold", markersize=10, label='Saturn')
    
    # Plot original velocity vectors
    ax1.quiver(x[0], x[1], x[2], v[0], v[1], v[2], color="black")
    
    # Plot original angular momentum vectors (scaled for visualization)
    angular_momentum = np.cross(x, v, axis=0) / 10**4
    ax1.quiver(x[0], x[1], x[2], angular_momentum[0], angular_momentum[1], angular_momentum[2], color="red", label='Angular Momentum (x10e-4)')

    # Plot center of mass for original data
    ax1.plot(center_of_mass[0], center_of_mass[1], center_of_mass[2], "x", color="black", markersize=10, label='Center of Mass')

    # Calculate center of mass for rotated data
    center_of_mass_rotated = calculate_center_of_mass(rotated_x, m)

    # Rotated data subplot
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_xlim((-2 * 10**8, 2 * 10**8))
    ax2.set_ylim((-2 * 10**8, 2 * 10**8))
    ax2.set_zlim((-2 * 10**8, 2 * 10**8))
    ax2.set_aspect('equal', adjustable='box')
    ax2.set_box_aspect(None, zoom=0.92)
    ax2.set_title("After Correction")
    
    # Axis labels
    ax2.set_xlabel('x (m)')
    ax2.set_ylabel('y (m)')
    ax2.set_zlabel('z (m)')

    # Plot rotated positions
    ax2.plot(rotated_x[0], rotated_x[1], rotated_x[2], ".", label='Moons')
    
    # Highlight Saturn in gold
    ax2.plot(rotated_x[0, saturn_index], rotated_x[1, saturn_index], rotated_x[2, saturn_index], "o", color="gold", markersize=10, label='Saturn')
    
    # Plot rotated velocity vectors
    ax2.quiver(rotated_x[0], rotated_x[1], rotated_x[2], rotated_v[0], rotated_v[1], rotated_v[2], color="black")
    
    # Plot rotated angular momentum vectors (scaled for visualization)
    rotated_angular_momentum = np.cross(rotated_x, rotated_v, axis=0) / 10**4
    ax2.quiver(rotated_x[0], rotated_x[1], rotated_x[2], rotated_angular_momentum[0], rotated_angular_momentum[1], rotated_angular_momentum[2], color="red", label='Angular Momentum (x10e-4)')

    # Plot center of mass for rotated data
    ax2.plot(center_of_mass_rotated[0], center_of_mass_rotated[1], center_of_mass_rotated[2], "x", color="black", markersize=10, label='Center of Mass')

    # Create a single legend
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right')

    # Adjust layout to ensure all labels are visible 
    plt.subplots_adjust(left=0.05, right=0.85, top=0.95, bottom=0.05, wspace=0.1)

    # Saving figure
    if path is not None:
        plt.savefig(path)

    plt.show()


def rotate_data(saturn_data_dict, epoch, show_plots=False, translate_to_center_of_mass_frame=False, path=None):
    """
    Rotate the data to be in the plane of Saturn's equator.

    Parameters:
        saturn_data_dict (dict): Dictionary of moon names and their data.
        epoch (list): Start Julian date or list of Julian dates.
        translate_to_center_of_mass_frame (bool): Whether to translate all points so that the center of mass is the origin.
        show_plots (bool): Whether to show a plot of the rotated vectors.
        path (str): Path to save the plot.

    Returns:
        dict: Rotated data dictionary.
    """
    # Reforming data into arrays
    x, v, m = create_arrays(saturn_data_dict)

    # Rotating data
    rotated_x, rotated_v = rotate_arrays(x, v, epoch)

    # Translating data to a center of mass frame
    if translate_to_center_of_mass_frame:
        rotated_x, rotated_v = translate_to_center_of_mass(rotated_x, rotated_v, m)

    # Reforming data into dictionary format
    rotated_data = arrays_to_dict(rotated_x, rotated_v, m, saturn_data_dict)

    # Plotting positions and velocity vectors before and after rotation
    if show_plots:
        saturn_index = list(saturn_data_dict.keys()).index("Saturn")
        plot_rotated_data(x, v, m, rotated_x, rotated_v, saturn_index, path)

    return rotated_data


def rotate_data_long(saturn_data_dict, epoch, show_plots=False, translate_to_center_of_mass_frame=False, path=None, return_dict=False):
    """
    Rotate the data to be in the plane of Saturn's equator for multiple timesteps.

    Parameters:
        saturn_data_dict (dict): Dictionary of moon names and their data.
        epoch (list): Start Julian date or list of Julian dates.
        translate_to_center_of_mass_frame (bool): Whether to translate all points so that the center of mass is the origin.
        show_plots (bool): Whether to show a plot of the rotated vectors.
        path (str): Path to save the plot.
        return_dict (bool): Whether to return the rotated data as a dictionary or arrays.

    Returns:
        dict or (numpy.ndarray, numpy.ndarray): Rotated data, either as a dictionary or arrays.
    """
    # Reforming data into arrays
    x, v, m = create_arrays_long(saturn_data_dict)
    
    # Loading Pan's data
    pan_dict = {"Saturn": {"ID": 699, "Mass": 5.6834e+26}, "Pan": {"ID": 618, "Mass": 4.95e+15}}
    saturn_data_pan = sms.get_saturn_moons_vectors(pan_dict, epoch)
    xp, vp, mp = create_arrays(saturn_data_pan)

    # Calculating Pan's angular momentum
    angular_momentum_pan = np.cross(xp, mp * vp, axis=0)[:, -1]

    # Rotating arrays
    rotation_matrix = rot_to_top(angular_momentum_pan)
    rotated_x_all = rotation_matrix @ x
    rotated_v_all = rotation_matrix @ v

    # Translating data to the center of mass frame
    if translate_to_center_of_mass_frame:
        center_of_mass_pos = calculate_center_of_mass(x, m)
        translated_x_all = rotated_x_all - center_of_mass_pos[:, np.newaxis]
        translated_v_all = rotated_v_all - center_of_mass_pos[:, np.newaxis]
    else:
        translated_x_all = rotated_x_all
        translated_v_all = rotated_v_all

    # Reforming data into dictionary format
    rotated_data = arrays_to_dict(translated_x_all, translated_v_all, m, saturn_data_dict)

    # Plotting positions and velocity vectors before and after rotation
    if show_plots:
        saturn_index = list(saturn_data_dict.keys()).index("Saturn")
        plot_rotated_data(x, v, m, translated_x_all, translated_v_all, saturn_index, path)

    if return_dict:
        return rotated_data
    else:
        return translated_x_all, translated_v_all
