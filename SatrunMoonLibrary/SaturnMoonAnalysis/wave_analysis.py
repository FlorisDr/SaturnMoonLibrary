import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde  # For KDE plotting

class WaveAnalysis:
    def __init__(self, dataset, bins=20, plot_type="2d"):
        self.dataset = dataset
        self.bins = bins
        self.s = int(np.ceil(np.sqrt(bins) / 4))
        self.plot_type = plot_type
        self.figure = None
        self.figure_3d = None
        self.axes = None
        self.lines = dict()
        self.edges = None
        self.dots = None  # For 3D plot elements
        self.animation_2d = None
        self.animation_3d = None
        self.setup_plots()

    def cartesian_to_polar(self, x, y):
        r = np.sqrt(x ** 2 + y ** 2)
        theta = np.arctan2(y, x)
        return r, np.mod(theta, 2 * np.pi)  # Ensure theta is in the range [0, 2pi]

    def split_theta_in_bins(self, r_test, theta_test, theta_min, theta_max, i, sieve):
        bindata_indexes = np.nonzero(np.logical_and(theta_min < theta_test, theta_test < theta_max))[0]
        return (
            r_test[bindata_indexes],
            self.dataset.relative_positions[i, self.dataset.num_moons:, 2][sieve][bindata_indexes] 
            - self.dataset.relative_positions[i, 0, 2],
            theta_test[bindata_indexes]
        )

    def setup_plots(self):
        """Sets up the plotting environment based on the plot type."""
        if self.plot_type in ["2d", "both"]:
            # 2D Plot Setup
            self.figure = plt.figure()
            self.axes = self.figure.subplots(self.s, self.s)

        if self.plot_type in ["3d", "both"]:
            # 3D Plot Setup
            self.figure_3d = plt.figure(figsize=(14, 7))
            self.ax_3d = self.figure_3d.add_subplot(121, projection='3d')

    def azimuthal_bin_analysis(self, i=0, r_min = 0.7e8, r_max= 1.5e8, z_min = 3600, z_max=3800):
        """Performs the azimuthal bin analysis and creates initial plots."""
        r_test, theta_test = self.cartesian_to_polar(
            self.dataset.relative_positions[i, self.dataset.num_moons:, 0],
            self.dataset.relative_positions[i, self.dataset.num_moons:, 1]
        )
        sieve = r_test < 1e11
        self.edges = np.histogram(theta_test, bins=self.bins)[1]

        if self.plot_type in ["2d", "both"]:
            # 2D Grid Plotting
            k = 0
            rlabel = True
            for axlist in self.axes:
                axlist[0].set_ylabel("z (m)")
                rlabel = not rlabel
                for ax in axlist:
                    ax.set_xlim(r_min, r_max)
                    ax.set_ylim(z_min,z_max)
                    if rlabel:
                        ax.set_xlabel(f"r (m)")
                    if k >= self.bins: 
                        break
                    r, z, _ = self.split_theta_in_bins(r_test[sieve], theta_test[sieve], self.edges[k], self.edges[k + 1], i, sieve)
                    self.lines[k] = ax.plot(r, z, "g.", markersize=2)[0]
                    k += 4
            plt.show()

        if self.plot_type in ["3d", "both"]:
            # 3D Plotting
            r, z, thet = self.split_theta_in_bins(r_test[sieve], theta_test[sieve], self.edges[0], self.edges[1], i, sieve)
            self.dots = self.ax_3d.plot(r, thet, z, ".", label="Test Particles", color="navy", markersize=.5)[0]
            self.ax_3d.set_zlim(-2e4, 0)
            self.ax_3d.set_xlim(.7e8, 2.4e8)
            plt.show()

    def azimuthal_bin_analysis_with_kde(self, i=0, r_min=0.7e8, r_max=1.5e8, z_min=3600, z_max=3800, bandwidth=None):
        """Performs azimuthal bin analysis and plots radial KDE for particles within z and r limits.
        Allows bandwidth adjustment for KDE smoothing."""

        # Convert cartesian coordinates to polar (r, theta)
        r_test, theta_test = self.cartesian_to_polar(
            self.dataset.relative_positions[i, self.dataset.num_moons:, 0],
            self.dataset.relative_positions[i, self.dataset.num_moons:, 1]
        )

        # Filter the particles based on r_min, r_max, z_min, z_max
        z_test = self.dataset.relative_positions[i, self.dataset.num_moons:, 2]
        valid_indices = (r_test >= r_min) & (r_test <= r_max) & (z_test >= z_min) & (z_test <= z_max)
        
        r_filtered = r_test[valid_indices]  # Radial distances of valid particles

        # KDE Calculation
        if len(r_filtered) > 0:
            # Apply bandwidth adjustment if specified
            kde = gaussian_kde(r_filtered, bw_method=bandwidth)
            r_grid = np.linspace(r_min, r_max, 100)  # Create grid over the radial range
            kde_values = kde(r_grid)

            # Plot the 1D KDE for radial density
            plt.plot(r_grid, kde_values, color='blue')
            plt.fill_between(r_grid, kde_values, color='blue', alpha=0.3)
            plt.title('Radial Density KDE')
            plt.xlabel('Radial Distance (r)')
            plt.ylabel('Density')
            plt.show()

            return r_grid, kde_values  # Return grid and KDE values for further analysis
        else:
            print("No valid particles within the specified limits.")
            return None, None

    def find_kde_peaks(self, r_grid, kde_values):
        """Finds peaks in the KDE values and calculates the average distance between these peaks."""
        peaks, _ = find_peaks(kde_values, height=0)  # Adjust 'height' based on your data characteristics

        if len(peaks) < 2:
            return {"average_peak_distance": None, "num_peaks": len(peaks)}

        # Calculate distances between consecutive peaks
        peak_distances = np.diff(r_grid[peaks])

        return {
            "average_peak_distance": np.mean(peak_distances) if len(peak_distances) > 0 else None,
            "num_peaks": len(peaks)
        }
    


    def plot_wavelength_over_time(self, r_min=None, r_max=None, z_min=3600, z_max=3800, bandwidth=None, xlim_low=0):
        """
        Plots the average peak distance over time using KDE, with horizontal lines showing the time-averaged values.

        Args:
            r_min: Minimum radial distance for wavelength calculation.
            r_max: Maximum radial distance for wavelength calculation.
            z_min: Minimum z value to consider.
            z_max: Maximum z value to consider.
            bandwidth: Bandwidth for KDE smoothing.
        """
        timesteps = np.arange(0, self.dataset.relative_positions.shape[0])
        avg_peak_distances = [0]

        # Calculate the average peak distances for each timestep
        for i in timesteps:
            if i == 0:
                continue
            r_grid, kde_values = self.azimuthal_bin_analysis_with_kde(i, r_min, r_max, z_min, z_max, bandwidth)
            if r_grid is not None and kde_values is not None:
                result = self.find_kde_peaks(r_grid, kde_values)
                avg_peak_distances.append(result["average_peak_distance"])

        # Compute the time-averaged values
        avg_peak_distance_time_avg = np.nanmean(avg_peak_distances[xlim_low:])  # Use nanmean to ignore NaN values
        avg_peak_distance_time_avg2 = np.nanmean(avg_peak_distances[10:])  # Use nanmean to ignore NaN values

        #timestep conversion:
        dt=self.dataset.header["dt"]*(self.dataset.header["Saved Points Modularity"])
        time = dt/60/60/24
        
        # Plotting
        plt.figure(figsize=(10, 6))
        plt.scatter(timesteps[xlim_low:]*time, avg_peak_distances[xlim_low:], label='Average Peak Distance', color='b', marker='o')

        # Set font size
        plt.rcParams.update({'font.size': 10})

        # Add horizontal line for time average
        plt.axhline(y=avg_peak_distance_time_avg, color='b', linestyle='--', label=f'Time Avg (Avg Peak Distance): {avg_peak_distance_time_avg:.2e}')
        plt.axhline(y=avg_peak_distance_time_avg2, color='r', linestyle='--', label=f'Time Avg (Avg Peak Distance after 10 time steps): {avg_peak_distance_time_avg2:.2e}')
        # Plot labels and title
        plt.xlabel('Time (Days)')
        plt.ylabel('Average Peak Distance (m)')

        # Set tick size
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        #plt.title(f"Average Peak Distance Over Time (r_min={r_min}, r_max={r_max}, z_min={z_min}, z_max={z_max})")
        plt.legend()
        plt.show()



