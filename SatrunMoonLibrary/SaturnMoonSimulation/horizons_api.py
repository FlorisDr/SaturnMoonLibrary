#import statements
from astroquery.jplhorizons import Horizons
from astropy.time import Time, TimeDelta

import numpy as np
import scipy.constants as sc

import math
import re
import warnings

#some usefull constants
AU= sc.au
G = sc.G

#there are some deprecation warnigs, since: AstropyDeprecationWarning: ``id_type``s 'majorbody' and 'id' are deprecated and replaced with ``None``, which has the same functionality. [astroquery.jplhorizons.core]
# to ignore these i put down the code line below, but still it should preferably be fixed.
warnings.filterwarnings('ignore', category=DeprecationWarning) #doesnt work dont know why.

def get_saturn_moons_masses(saturn_data, epoch):
    """
    Warning: not functional
    Get masses for Saturn's moons and update the input dictionary with these values.
    Note: if the Mass from Horizons is somehow not in g or kg it won't find it.

    Parameters:
        saturn_data (dict): Dictionary of moon names and their IDs.
        epoch (list): Julian date(s) or list of calendar dates to query.
        
    Returns:
        dict: Updated dictionary with moon names as keys and their masses added.
    """
    center_id = '500@6'  # Center of the system is Saturn
    
    for moon, data in saturn_data.items():
        target_id = data["ID"]
        # Query the Horizons database for each moon
        moon_data = Horizons(id=target_id, location=center_id, epochs=epoch, id_type=None)
        response = moon_data.ephemerides_async(get_raw_response=True).text
        
        # Extract mass information from the response
        mass_info = re.findall(r"Mass \(10\^(\d\d?) (kg|g)\s?\s?\)\s+=\s+([\d.]+)", response)
        if mass_info:
            e, u, m = mass_info[0]
            mass = float(m) * 10**int(e) * 10**(3*(1-("k" in u)))
        else:
            mass = np.nan  # If no mass information is found, set mass to NaN
        
        # Update the dictionary with the mass
        saturn_data[moon]["Mass"] = mass
    
    return saturn_data

def get_saturn_moons_radii(saturn_data_dict, epoch):
    """
    Get radii of Saturn's moons.
   
    Parameters:
        saturn_moons_ids (dict): Dictionary of moon names and their Horizons IDs.
        epoch (list): Julian date(s) or list of calendar dates to query.
       
    Returns:
        dict: Nested dictionary with moon names as keys and radius values.
    """
    results = saturn_data_dict.copy()
    center_id = '500@6'  # Center of the system is Saturn
    
    for moon, data in saturn_data_dict.items():
        # Query the Horizons database for each moon
        moon_data = Horizons(id=data["ID"], location=center_id, epochs=epoch, id_type='id')
        response = moon_data.ephemerides_async(get_raw_response=True).text
        
        if moon == "Daphnis":
            results[moon]["Radius"] = 4.6e3
            continue  # Manually looked up, as it lists radii rather than mean radius
        
        elif moon == "Saturn":
            results[moon]["Radius"] = 58232e3
            continue  # This one puts the +- after the number without space
        
        r_data = response.split("Radius")[1].split("Density")[0]  # Looks between Radius and Density
        r_dat = r_data.split("=")[1].split(" ")[1]  # Value is after the equals
        
        if "x" in r_dat:
            # If the database lists 3 radii, grab the largest (first one as listed in descending order)
            results[moon]["Radius"] = float(r_dat.split("x")[0]) * 1e3
        else:
            results[moon]["Radius"] = float(r_dat) * 1e3
    
    return results

def get_saturn_moons_data(saturn_moons_data, epoch = [2458075.5], semi_major=True, include_eccentricity=True, include_inclination=True, units=True):
    """
    Get orbital elements (semi-major axis, eccentricity, and inclination) for Saturn's moons relative to Saturn.
    
    Parameters:
        saturn_moons_data (dict): Dictionary of moon names and their Horizons IDs.
        epoch (list): Julian date(s) or list of calendar dates to query. Default is [2458075.5] (then Daphnis will not Error)
        semi_major (bool): Whether to include the semi-major axis in the output. Default is True.
        include_eccentricity (bool): Whether to include eccentricity in the output. Default is True.
        include_inclination (bool): Whether to include inclination in the output. Default is True.
        units (bool): Whether to use SI units (True) or AU (False). Default is True.
        
    Returns:
        dict: Nested dictionary with moon names as keys and their orbital elements as values.
    """
    results = {}
    center_id = '500@6'  # Center of the system is Saturn
    
    for moon, data in saturn_moons_data.items():
        target_id = data["ID"]
        # Query the Horizons database for each moon
        moon_data = Horizons(id=target_id, location=center_id, epochs=epoch, id_type='id')
        elements = moon_data.elements()
        
        # Initialize result dictionary with existing keys and values
        result = data.copy()
        
        # Extract and convert orbital elements
        if semi_major:
            a = float(elements["a"])
            if units:
                a *= AU
            result["semi-major axis"] = a
        
        if include_eccentricity:
            result["eccentricity"] = float(elements["e"])
        
        if include_inclination:
            result["inclination"] = float(elements["incl"])
        
        # Add to the results dictionary
        results[moon] = result
    
    return results

def create_epoch_dict(epoch, dt, number_of_timesteps, number_of_saved_points):
    """
    Creates a dictionary with 'start', 'stop', and 'step' keys based on the given parameters.
    Takes into account the number_of_saved_points to ensure the step is adjusted accordingly.
    
    Parameters:
        epoch (float): Start time in Julian date format.
        dt (float): Time step spacing in seconds.
        number_of_timesteps (int): Total number of timesteps.
        number_of_saved_points (int): Number of points to save from the total timesteps.
        
    Returns:
        dict: Dictionary with 'start', 'stop', and 'step' keys.
    """
    # Convert the start epoch (Julian date) to ISO format for human-readable dates
    start_time = Time(epoch, format='jd')
    
    # Calculate the total duration in seconds
    total_duration_seconds = dt * (number_of_timesteps - 1)
    
    # converting dt
    step_in_minutes = dt // 60

    if dt%60 != 0:
        print(f"dt = {dt} is not perfectly divissible")

    # Convert the total duration to days and add to the start time to get the stop time
    total_duration_days = total_duration_seconds / 86400  # Convert seconds to days
    stop_time = start_time + TimeDelta(total_duration_days, format='jd')


    # Create the epoch dictionary with the step in seconds (ending with 's')
    # Modify the epoch dictionary creation
    epoch_dict = {
        "start": start_time.iso[0].split('.')[0],  # Remove milliseconds
        "stop": stop_time.iso[0].split('.')[0],    # Remove milliseconds
        "step": f"{int(step_in_minutes)}m"       # Adjusted step size for saved points, in seconds
    }

    return epoch_dict

def get_saturn_moons_vectors(saturn_data_dict, epoch, dt=None, number_of_timesteps=1, number_of_saved_points=1, units=True, include_time=False):
    """
    Get position and velocity vectors for Saturn's moons relative to Saturn over one or multiple timesteps.
    
    Parameters:
        saturn_data_dict (dict): Dictionary of moon names and their data.
        epoch (list): Start Julian date or list of Julian dates.
        dt (float, optional): Time step spacing in seconds. Default is None.
        number_of_timesteps (int, optional): Total number of timesteps. Default is 1.
        number_of_saved_points (int, optional): Number of points to save from the generated timesteps. Default is 1.
        units (bool): Whether to use SI units (True) or AU (False). Default is True.
        include_time (bool): Whether to include the t_i tuple in the output. Default is False.
        
    Returns:
        dict: Nested dictionary with moon names as keys and updated data including position/velocity vectors.
    """
    results = {}
    center_id = '500@6'  # Center of the system is Saturn

    if number_of_timesteps > 1 and dt:
        # Generate the array of timesteps and select saved points
        timesteps = np.linspace(0, dt * (number_of_timesteps - 1), number_of_timesteps) / 86400  # Convert seconds to days
        saved_indices = np.round(np.linspace(0, number_of_timesteps - 1, number_of_saved_points)).astype(int)
        saved_timesteps = epoch + timesteps[saved_indices]
        epoch_dict = create_epoch_dict(epoch, dt, number_of_timesteps, number_of_saved_points)
    else:
        saved_timesteps = epoch_dict = epoch
        saved_indices = [0]

    for moon, data in saturn_data_dict.items():
        target_id = data["ID"]
        updated_data = data.copy()

        try:
            # Query the Horizons database for the moon across all saved timesteps
            moon_data = Horizons(id=target_id, location=center_id, epochs= epoch_dict, id_type=None)
            vectors = moon_data.vectors()

            # Extract and save position, velocity vectors, and time for each saved timestep
            for i, index in enumerate(saved_indices):
                try:
                    if units:
                        position = {
                            f"r_{i}": {
                                "x": float(vectors['x'][i] * AU),
                                "y": float(vectors['y'][i] * AU),
                                "z": float(vectors['z'][i] * AU)
                            }
                        }
                        velocity = {
                            f"v_{i}": {
                                "vx": float(vectors['vx'][i] * AU / 24 / 60 / 60),
                                "vy": float(vectors['vy'][i] * AU / 24 / 60 / 60),
                                "vz": float(vectors['vz'][i] * AU / 24 / 60 / 60)
                            }
                        }
                    else:
                        position = {
                            f"r_{i}": {
                                "x": float(vectors['x'][i]),
                                "y": float(vectors['y'][i]),
                                "z": float(vectors['z'][i])
                            }
                        }
                        velocity = {
                            f"v_{i}": {
                                "vx": float(vectors['vx'][i]),
                                "vy": float(vectors['vy'][i]),
                                "vz": float(vectors['vz'][i])
                            }
                        }
                    
                    if include_time:
                        elapsed_seconds = saved_indices[i] * (dt if dt else 0)
                        time = {f"t_{i}": (saved_timesteps[i], elapsed_seconds)}
                        updated_data.update(time)

                    # Update the dictionary with indexed r_i and v_i
                    updated_data.update(position)
                    updated_data.update(velocity)

                except Exception:
                    # Handle missing data for specific timesteps
                    updated_data[f"r_{i}"] = {"x": math.nan, "y": math.nan, "z": math.nan}
                    updated_data[f"v_{i}"] = {"vx": math.nan, "vy": math.nan, "vz": math.nan}
                    if include_time:
                        updated_data[f"t_{i}"] = math.nan

        except Exception as e:
            # Handle general errors in querying data for a moon
            print(f"Error querying Horizons for {moon}: {str(e)}")
            for i in range(number_of_saved_points):
                updated_data[f"r_{i}"] = {"x": math.nan, "y": math.nan, "z": math.nan}
                updated_data[f"v_{i}"] = {"vx": math.nan, "vy": math.nan, "vz": math.nan}
                if include_time:
                    updated_data[f"t_{i}"] = math.nan

        # Add to the results dictionary
        results[moon] = updated_data

    return results
