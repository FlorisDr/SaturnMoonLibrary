import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np

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

    def azimuthal_bin_analysis(self, i=0, r_min = 0.7e8, r_max= 1.5e8):
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

    def update_2d(self, i):
        """Updates 2D plot for a new frame."""
        k = 0
        for axlist in self.axes:
            for ax in axlist:
                if k >= self.bins: 
                    break
                r_test, theta_test = self.cartesian_to_polar(
                    self.dataset.relative_positions[i, self.dataset.num_moons:, 0] - self.dataset.relative_positions[i, 0, 0],
                    self.dataset.relative_positions[i, self.dataset.num_moons:, 1] - self.dataset.relative_positions[i, 0, 1]
                )
                sieve = r_test < 1e11
                r, z, _ = self.split_theta_in_bins(r_test[sieve], theta_test[sieve], self.edges[k], self.edges[k + 1], i, sieve)
                self.lines[k].set_data(r, z)
                k += 4

    def update_3d(self, i):
        """Updates 3D plot for a new frame."""
        r_test, theta_test = self.cartesian_to_polar(
            self.dataset.relative_positions[i, self.dataset.num_moons:, 0],
            self.dataset.relative_positions[i, self.dataset.num_moons:, 1]
        )
        sieve = r_test < 1e11
        r, z, thet = self.split_theta_in_bins(r_test[sieve], theta_test[sieve], self.edges[0], self.edges[1], i, sieve)
        self.dots.set_data_3d(r, thet, z)

    def animate(self):
        """Runs the animation based on the selected plot type."""
        # Ensure that the animations are properly stored to avoid garbage collection
        if self.plot_type in ["2d", "both"]:
            self.animation_2d = anim.FuncAnimation(self.figure, self.update_2d, 
                                                   frames=np.arange(0, self.dataset.relative_positions.shape[0], 1),
                                                   interval=10)
        if self.plot_type in ["3d", "both"]:
            self.animation_3d = anim.FuncAnimation(self.figure_3d, self.update_3d, 
                                                   frames=np.arange(0, self.dataset.relative_positions.shape[0], 1),
                                                   interval=10)

        # Ensure animations are retained and shown properly
        if self.plot_type == "2d":
            plt.show()  # Show 2D animation
        elif self.plot_type == "3d":
            plt.show()  # Show 3D animation
        elif self.plot_type == "both":
            plt.show()  # Ensure both 2D and 3D animations are shown

        return self.animation_2d, self.animation_3d  # Return animations to prevent garbage collection
    
    
    def compute_wavelength_2d(self, i=0, r_min=None, r_max=None):
        """
        Computes the average and median wavelength from the 2D radial distance (r) for the z-displacements.
        
        Args:
            i: The index of the timestep to analyze.
            r_min: Minimum value of r to consider.
            r_max: Maximum value of r to consider.

        Returns:
            dict: Dictionary containing the average wavelength, median wavelength, and number of peaks found.
        """
        r_test, theta_test = self.cartesian_to_polar(
            self.dataset.relative_positions[i, self.dataset.num_moons:, 0],
            self.dataset.relative_positions[i, self.dataset.num_moons:, 1]
        )
        sieve = r_test < 1e11
        r, z, _ = self.split_theta_in_bins(r_test[sieve], theta_test[sieve], self.edges[0], self.edges[1], i, sieve)
        
        # Filter by r_min and r_max
        if r_min is not None:
            r = r[r >= r_min]
            z = z[len(z) - len(r):]  # Ensure z corresponds to the filtered r values
        if r_max is not None:
            r = r[r <= r_max]
            z = z[:len(r)]  # Ensure z corresponds to the filtered r values

        # Find peaks in the z-displacement data
        peaks = np.where(np.diff(np.sign(np.diff(z))) < 0)[0]  # Find local maxima in z
        if len(peaks) < 2:
            return {"average_wavelength": None, "medianwavelength": None, "num_peaks": 0}  # Not enough peaks to calculate wavelength
        
        # Calculate wavelength as the absolute distance between peaks in r
        wavelengths = np.abs(np.diff(r[peaks]))
        return {
            "average_wavelength": np.mean(wavelengths) if len(wavelengths) > 0 else None,
            "medianwavelength": np.median(wavelengths) if len(wavelengths) > 0 else None,
            "num_peaks": len(peaks)
        }

    def plot_wavelength_over_time(self, r_min=None, r_max=None):
        """
        Plots the average and median wavelength over time, with horizontal lines showing their time-averaged values.

        Args:
            r_min: Minimum radial distance for wavelength calculation.
            r_max: Maximum radial distance for wavelength calculation.
        """
        timesteps = np.arange(0, self.dataset.relative_positions.shape[0])
        avg_wavelengths = []
        medianwavelengths = []

        # Calculate the average and median wavelengths for each timestep
        for i in timesteps:
            result = self.compute_wavelength_2d(i, r_min, r_max)
            avg_wavelengths.append(result["average_wavelength"])
            medianwavelengths.append(result["medianwavelength"])

        # Compute the time-averaged values
        avg_wavelength_time_avg = np.nanmean(avg_wavelengths)  # Use nanmean to ignore NaN values
        medianwavelength_time_avg = np.nanmean(medianwavelengths)

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.scatter(timesteps, avg_wavelengths, label='Average Wavelength', color='b', marker='o')
        plt.scatter(timesteps, medianwavelengths, label='Median Wavelength', color='g', marker='x')

        # Add horizontal lines for time averages
        plt.axhline(y=avg_wavelength_time_avg, color='b', linestyle='--', label=f'Time Avg (Avg Wavelength): {avg_wavelength_time_avg:.2f}')
        plt.axhline(y=medianwavelength_time_avg, color='g', linestyle='--', label=f'Time Avg (Median Wavelength): {medianwavelength_time_avg:.2f}')

        # Plot labels and title
        plt.xlabel('Timestep')
        plt.ylabel('Wavelength')
        plt.title(f"Wavelength Over Time (r_min={r_min}, r_max={r_max})")
        plt.legend()
        plt.show()


