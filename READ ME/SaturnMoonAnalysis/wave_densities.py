import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

import math

class WaveDensity:
    def __init__(self, database):
        """
        Initializes the WaveDensity class with a dataset.

        Parameters:
        - database: The dataset containing particle positions over time.
        """
        self.database = database

    def plot_radial_density(self, timestep, bins=50, r_min=7e7, r_max=1.4e8):
        """
        Plots a histogram of the radial density of test particles at a specific timestep.

        Parameters:
        - timestep: The index of the timestep to analyze.
        - bins: Number of bins in the histogram (default = 50).
        - r_min: Minimum radial distance to plot (default = 7e7).
        - r_max: Maximum radial distance to plot (default = 1.4e8).
        """
        if timestep < 0 or timestep >= self.database.positions.shape[0]:
            print(f"⚠ WARNING: Timestep {timestep} is out of range.")
            return

        # Extract positions at the given timestep
        positions = self.database.positions[timestep]

        # Compute radial distances
        radial_distances = np.sqrt(np.sum(positions**2, axis=1))

        # Create histogram plot
        plt.figure(figsize=(8, 5))
        plt.hist(radial_distances, bins=bins, range=(r_min, r_max), edgecolor='black', density=True)
        plt.xlabel('Radial Distance')
        plt.ylabel('Density')
        plt.title(f'Radial Density of Test Particles (Timestep {timestep})')
        plt.grid(True)
        plt.show()

    def plot_kde_density(self, timestep, bandwidth=1e6, r_min=6.8e7, r_max=1.43e8, num_samples=1000):
        """
        Plots the Kernel Density Estimate (KDE) of the radial density of test particles at a specific timestep.

        Parameters:
        - timestep: The index of the timestep to analyze.
        - bandwidth: The bandwidth parameter for KDE (default = 1e6).
        - r_min: Minimum radial distance to plot (default = 7e7).
        - r_max: Maximum radial distance to plot (default = 1.4e8).
        - num_samples: Number of points in the KDE evaluation grid (default = 1000).
        """
        if timestep < 0 or timestep >= self.database.positions.shape[0]:
            print(f"⚠ WARNING: Timestep {timestep} is out of range.")
            return

        # Extract positions at the given timestep
        positions = self.database.positions[timestep]

        # Compute radial distances
        radial_distances = np.sqrt(np.sum(positions**2, axis=1)).reshape(-1, 1)  # Reshape for sklearn

        # Fit KDE model
        kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
        kde.fit(radial_distances)

        # Define range of values for density estimation
        r_values = np.linspace(r_min, r_max, num_samples).reshape(-1, 1)

        # Evaluate KDE on the grid
        log_density = kde.score_samples(r_values)  # Get log-density
        density = np.exp(log_density)  # Convert back from log-scale

        # Plot KDE density
        plt.figure(figsize=(8, 5))
        plt.plot(r_values, density, label="KDE Density", color="blue")
        plt.xlabel('Radial Distance')
        plt.ylabel('Density')
        plt.title(f'Radial KDE Density of Test Particles (Timestep {timestep})')
        plt.grid(True)
        plt.legend()
        plt.show()


    def plot_mosaic(self, timesteps, bins=50, r_min=7e7, r_max=1.4e8):
        """
        Plots a mosaic grid of radial density histograms for multiple timesteps.

        Parameters:
        - timesteps: A list of timestep indices to visualize.
        - bins: Number of bins in the histograms.
        - r_min: Minimum radial distance.
        - r_max: Maximum radial distance.
        """
        num_plots = len(timesteps)
        if num_plots == 0:
            print("⚠ WARNING: No timesteps provided for mosaic plot.")
            return

        # Determine grid size (closest to square)
        grid_size = math.ceil(math.sqrt(num_plots))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 4, grid_size * 4))

        # Flatten axes array if needed
        axes = np.array(axes).flatten()

        for i, timestep in enumerate(timesteps):
            if timestep < 0 or timestep >= self.database.positions.shape[0]:
                print(f"⚠ WARNING: Timestep {timestep} is out of range, skipping.")
                continue

            # Compute radial distances
            positions = self.database.positions[timestep]
            radial_distances = np.sqrt(np.sum(positions**2, axis=1))

            # Plot histogram in the corresponding subplot
            axes[i].hist(radial_distances, bins=bins, range=(r_min, r_max), edgecolor='black', density=True)
            axes[i].set_title(f"Timestep {timestep}")
            axes[i].set_xlabel("Radial Distance")
            axes[i].set_ylabel("Density")
            axes[i].grid(True)

        # Hide unused subplots if grid is larger than needed
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

    # def plot_mosaic(self, timesteps, plot_func, figsize=(10, 10), **kwargs):
    #     """
    #     Plots a mosaic of radial density plots (histogram or KDE) for multiple timesteps.

    #     Parameters:
    #     - timesteps: List of timesteps to plot.
    #     - plot_func: The function to use for plotting (e.g., self.plot_radial_histogram or self.plot_kde_density).
    #     - figsize: Tuple specifying the figure size (default: (10, 10)).
    #     - **kwargs: Additional arguments to pass to the plotting function.
    #     """
    #     num_plots = len(timesteps)
        
    #     # Determine the most square-like grid layout
    #     rows = math.floor(math.sqrt(num_plots))
    #     cols = math.ceil(num_plots / rows)

    #     fig, axes = plt.subplots(rows, cols, figsize=figsize)
    #     axes = np.array(axes).flatten()  # Flatten in case of odd numbers

    #     for i, timestep in enumerate(timesteps):
    #         plt.sca(axes[i])  # Set current axis
    #         plot_func(timestep, **kwargs)  # Call the function with kwargs
    #         axes[i].set_title(f'Timestep {timestep}')  # Label each plot

    #     # Hide any unused subplots
    #     for j in range(i + 1, len(axes)):
    #         fig.delaxes(axes[j])

    #     plt.tight_layout()
    #     plt.show()



