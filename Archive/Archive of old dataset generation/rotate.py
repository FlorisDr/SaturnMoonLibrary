#import statements
import numpy as np
import matplotlib.pyplot as plt

import SaturnMoonLibrary.SaturnMoonSimulation as sms

# Function to rotate a vector to align with the z-axis
def rot_to_top(h):
    x, y, z = h / np.sqrt(np.dot(h, h))
    s, r = np.sqrt(x**2 + y**2), np.sqrt(x**2 + y**2 + z**2)

    if x == y == 0:
        return np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    return np.matmul(
        np.array([[z / r, 0, -s / r], [0, 1, 0], [s / r, 0, z / r]]),
        np.array([[x / s, y / s, 0], [-y / s, x / s, 0], [0, 0, 1]])
    )

# Function to create position, velocity, and mass arrays from the data dictionary
def create_arrays(data_dict):
    x = np.zeros((3, len(data_dict)))
    v = np.zeros((3, len(data_dict)))
    m = np.zeros(len(data_dict))
    for i, (key, item) in enumerate(data_dict.items()):
        for k, coord in zip(range(3), ["x", "y", "z"]):
            x[k, i] = item["r_0"][coord] - data_dict["Saturn"]["r_0"][coord]
            v[k, i] = item["v_0"]["v" + coord] - data_dict["Saturn"]["v_0"]["v" + coord]
        m[i] = item["Mass"]
    return x, v, m

# Function to rotate position and velocity arrays
def rotate_arrays(x, v, epoch):
    
    #Since Pan has an inclination of almost 0 degrees, 
    #we will use Pan's angular momentum vector to rotate the axes such that the x-y plane lies true Saturns equator
    
    # loading in the Data of Pan
    pan_dict ={"Saturn": {"ID": 699, "Mass": 5.6834e+26},"Pan": {"ID": 618, "Mass": 4.95e+15}}
    saturn_data_pan = sms.get_saturn_moons_vectors(pan_dict, epoch)
    xp,vp,mp = create_arrays(saturn_data_pan)

    #caluclating the angular momentum of Pan
    angular_momentum_pan = np.cross(xp,mp*vp,axis = 0)[:,-1]

    # Rotating arrays
    rotation_matrix = rot_to_top(angular_momentum_pan)
    rotated_x = rotation_matrix @ x
    rotated_v = rotation_matrix @ v

    return rotated_x, rotated_v

# Function to transform arrays back into the dictionary format
def arrays_to_dict(x, v, m, data_dict):
    transformed_dict = {}
    keys = list(data_dict.keys())

    for i, key in enumerate(keys):
        transformed_dict[key] = {}
        
        # Copy existing keys and values
        for k, value in data_dict[key].items():
            transformed_dict[key][k] = value
        
        # Add mass
        transformed_dict[key]["Mass"] = m[i]
        
        # Add r_0 vector if x is array-like and has sufficient dimensions
        if isinstance(x, np.ndarray) and x.ndim == 2:
            transformed_dict[key]["r_0"] = {
                "x": x[0, i],
                "y": x[1, i],
                "z": x[2, i]
            }
        
        # Add v_0 vector if v is array-like and has sufficient dimensions
        if isinstance(v, np.ndarray) and v.ndim == 2:
            transformed_dict[key]["v_0"] = {
                "vx": v[0, i],
                "vy": v[1, i],
                "vz": v[2, i]
            }

    return transformed_dict

def calculate_center_of_mass(x, m):
    """
    Calculate the center of mass position.
    
    Parameters:
        x (numpy.ndarray): Array of shape (3, N) containing position vectors.
        m (numpy.ndarray): Array of shape (N,) containing masses.
        
    Returns:
        numpy.ndarray: Center of mass position.
    """
    total_mass = np.sum(m)
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
    """
    # Calculate the center of mass position
    center_of_mass_pos = calculate_center_of_mass(x, m)
    
    # Translate positions
    translated_x = x - center_of_mass_pos[:, np.newaxis]

    # Total mass
    total_mass = np.sum(m)
    
    # Calculate the center of mass velocity
    center_of_mass_vel = np.sum(v * m, axis=1) / total_mass
    
    # Translate velocities
    translated_v = v - center_of_mass_vel[:, np.newaxis]
    
    return translated_x, translated_v

# Function to plot the data before rotation and after rotation
def plot_rotated_data(x, v, m, rotated_x, rotated_v, saturn_index=0, path = None):
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



# Function that chains the rotation fucntions
def rotate_data(saturn_data_dict, epoch, show_plots = False, translate_to_center_of_mass_frame = False, path = None):
    """
    Rotating the data to be in the plane of Saturns equator

    Parameters:
    saturn_data_dict (dict): Dictionary of moon names and their data.
    epoch (list): Start Julian date or list of Julian dates.
    translate_to_center_of_mass_frame (bool): Wheter to to translate all points sucht that the center of mass is the origin
    show_plots (bool): Wheter to show a plot of the rotated vectors.
    """
    # Reformating data to arrays
    x, v, m = create_arrays(saturn_data_dict)

    # Rotating data
    rotated_x, rotated_v = rotate_arrays(x, v, epoch)

    # Translating data to a center of mass frame
    if translate_to_center_of_mass_frame:
        rotated_x, rotated_v = translate_to_center_of_mass(rotated_x, rotated_v, m)

    # Reformating data to dictionary format
    rotated_data = arrays_to_dict(rotated_x, rotated_v, m, saturn_data_dict)

    # Plotting positions and velocity vectors before and after rotation
    if show_plots:
        saturn_index = list(saturn_data_dict.keys()).index("Saturn")
        plot_rotated_data(x, v, m, rotated_x, rotated_v, saturn_index, path)

    return rotated_data

# def rotate_data(saturn_data_dict, epoch, show_plots=False, translate_to_center_of_mass_frame=False, path=None):
#     """
#     Rotating the data to be in the plane of Saturn's equator

#     Parameters:
#     saturn_data_dict (dict): Dictionary of moon names and their data.
#     epoch (list): Start Julian date or list of Julian dates.
#     translate_to_center_of_mass_frame (bool): Whether to translate all points such that the center of mass is the origin.
#     show_plots (bool): Whether to show a plot of the rotated vectors.
#     """
#     # Reformating data to arrays
#     x, v, m = create_arrays(saturn_data_dict)

#     # Initialize a list to store rotated data for each timestep
#     rotated_x_all = []
#     rotated_v_all = []

#     # Apply rotation for each timestep
#     for timestep in range(len(epoch)):
#         # Get the data for the specific timestep
#         rotated_x, rotated_v = rotate_arrays(x, v, epoch[timestep])

#         rotated_x_all.append(rotated_x)
#         rotated_v_all.append(rotated_v)

#     # Stack all timesteps into arrays
#     rotated_x_all = np.array(rotated_x_all)
#     rotated_v_all = np.array(rotated_v_all)

#     # Calculate center of mass for the reference timestep (t_0)
#     if translate_to_center_of_mass_frame:
#         # Use the first timestep (t_0) to calculate the center of mass
#         center_of_mass_pos = calculate_center_of_mass(x, m)

#         # Translate all timesteps to the center of mass frame
#         translated_x_all = rotated_x_all - center_of_mass_pos[:, np.newaxis]
#         translated_v_all = rotated_v_all - center_of_mass_pos[:, np.newaxis]
#     else:
#         translated_x_all = rotated_x_all
#         translated_v_all = rotated_v_all

#     # Reformating data to dictionary format
#     rotated_data = arrays_to_dict(translated_x_all, translated_v_all, m, saturn_data_dict)

#     # Plotting positions and velocity vectors before and after rotation
#     if show_plots:
#         saturn_index = list(saturn_data_dict.keys()).index("Saturn")
#         plot_rotated_data(x, v, m, translated_x_all, translated_v_all, saturn_index, path)

#     return rotated_data
