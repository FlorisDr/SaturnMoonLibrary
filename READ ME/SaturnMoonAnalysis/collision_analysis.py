import numpy as np
import matplotlib.pyplot as plt

class CollisionAnalysis:
    def __init__(self, database):
        self.database = database
        self.pre_collision_positions = np.array([])  # Initialize as an empty NumPy array

    def collisions_count(self):
        # Calculate the distance of each particle from the origin
        dt = self.database.header["dt"] * self.database.header["Saved Points Modularity"]

        # Identify particles that have been displaced to (1e12, 1e12, 1e12) + original position
        displaced_vector = np.array([1e12, 1e12, 1e12,0,0,0])
        displaced_mask = np.logical_and(self.database.positions[:,:,0] >= 7.5e11,np.all(self.database.positions[:,:,3:] ==0,axis=2))  # Check if all positions are at 1e12 or more
        # If all particles are displaced, warn the user
        if np.all(self.database.positions[-1,displaced_mask[-1,:],:3]==1e12):
            print("⚠ WARNING: All collision particles are at the displacement vector (1e12, 1e12, 1e12).")
            print("This suggests the dataset might be outdated, and original positions may not be recoverable.")
            return  # Exit the function without updating pre_collision_positions

        # Recover the original position before displacement
        self.pre_collision_positions = self.database.positions[-1,displaced_mask[-1,:],:] - displaced_vector

        # Count the number of displaced particles over time
        displaced_mask = np.logical_and(self.database.positions[:,:,0] >= 7.5e11,np.all(self.database.positions[:,:,3:] ==0,axis=2))  # Check if all positions are at 1e12 or more
        displaced_particles_count = np.sum(displaced_mask, axis=1)
        
        # Plot the count of displaced particles over time
        plt.figure(figsize=(10, 5))
        
        # Set font size before any plotting calls
        SMALL_SIZE = 8
        MEDIUM_SIZE = 10
        BIGGER_SIZE = 12

        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=15)     # fontsize of the axes title
        plt.rc('axes', labelsize=15)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=12)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize

        plt.plot(dt / 60 / 60 / 24 * np.arange(len(displaced_particles_count)), displaced_particles_count, label='Number of collisions')
        plt.xlabel('Time (days)')
        plt.ylabel('Count')
        plt.legend()
        plt.show()


    def plot_radial_histogram(self, bins=30, r_min=0.7e8, r_max=1.4e8):
        if self.pre_collision_positions.size == 0:
            print("No pre-collision positions stored. Please run collisions_count() first.")
            return

        # Calculate the radial distances of recovered pre-collision positions
        radial_distances = np.sqrt(np.sum(self.pre_collision_positions ** 2, axis=1))

        # Create figure
        plt.figure(figsize=(10, 5))
        plt.hist(radial_distances, bins=bins, edgecolor='black', label="Collisions")

        # Close moons to highlight
        close_moons = ["Daphnis", "Atlas", "Pandora", "Pan", "Prometheus"]
        moon_data = {moon.name: moon for moon in self.database.moons}

        # Dictionary of moon radii in meters (if known)
        moon_radii = {
            "Daphnis": 4.6e3,  # meters
            "Mimas": 198.8e3,
            "Janus": 101.7e3,
            "Epimetheus": 64.9e3,
            "Atlas": 20.5e3,
            "Pandora": 52.2e3,
            "Pan": 17.2e3,
            "Prometheus": 68.8e3,
            "Enceladus": 252.3e3
        }

        patches = []  # For legend entries

        for moon in close_moons:
            if moon in moon_data:
                moon_obj = moon_data[moon]
                orbital_params, _ = moon_obj.calculate_orbital_elements()

                semi_major_axis = np.mean(orbital_params["semimajor_axis"])  # Ensure scalar
                eccentricity = np.mean(orbital_params["eccentricity"])  # Ensure scalar

                # Compute min and max radius from semi-major axis and eccentricity
                r_min_moon = semi_major_axis * (1 - eccentricity)
                r_max_moon = semi_major_axis * (1 + eccentricity)

                # Add the moon's physical radius (if known)
                if moon in moon_radii:
                    radius = moon_radii[moon]
                    r_min_moon -= radius
                    r_max_moon += radius

                # Plot shaded region
                patch = plt.axvspan(float(r_min_moon), float(r_max_moon), color=moon_obj.color, alpha=0.3, label=moon)
                patches.append(patch)

        # Format plot
        plt.xlim(r_min, r_max)
        plt.xlabel('Radial Distance: r(m)')
        plt.ylabel('Frequency')
        SMALL_SIZE = 8
        MEDIUM_SIZE = 10
        BIGGER_SIZE = 12

        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=15)     # fontsize of the axes title
        plt.rc('axes', labelsize=15)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=12)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize
        #plt.title('Radial Histogram of Collisions')

        # Place legend outside
        plt.legend(handles=patches, loc='upper left', bbox_to_anchor=(1, 1))
        plt.show()


