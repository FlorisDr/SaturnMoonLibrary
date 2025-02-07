import numpy as np
import matplotlib.pyplot as plt

class WaveStills:
    def __init__(self, dataset):
        self.dataset = dataset
    
    def plot_polar_cartesian_still(self, timestep, r_max, r_min=0, ax=None, show=True, theta_max=2*np.pi, heatmap=False):
        """Generates a single frame of the polar to Cartesian plot at a given timestep."""
        if ax is None:
            figure, ax = plt.subplots(figsize=(10, 7))  # Create new figure if no axis is provided

        ax.set_xlim(r_min, r_max)
        ax.set_ylim(0, theta_max)
        ax.set_xlabel('Radial Distance (r) [m]')
        ax.set_ylabel('Angle (theta) [radians]')

        def cartesian_to_polar(x, y):
            r = np.sqrt(x ** 2 + y ** 2)
            theta = np.arctan2(y, x)
            return r, np.mod(theta, 2 * np.pi)

        # Store legend handles
        legend_handles = []
        for body in self.dataset.moons:
            r, theta = cartesian_to_polar(body.pos[timestep, 0], body.pos[timestep, 1])
            marker, = ax.plot(r, theta, 'o', color=body.color, label=body.name)
            legend_handles.append(marker)

        if not heatmap:
            r_test, theta_test = cartesian_to_polar(
                self.dataset.relative_positions[timestep, self.dataset.num_moons:, 0],
                self.dataset.relative_positions[timestep, self.dataset.num_moons:, 1]
            )
            test_marker, = ax.plot(r_test, theta_test, '.', color="navy", markersize=1, label="Test Particles")
            legend_handles.append(test_marker)

        saturn_marker, = ax.plot([0], [0], "x", color="yellow", markersize=10, label="Saturn (Barycenter)")
        legend_handles.append(saturn_marker)

        ax.set_title(f"Polar Plot at timestep {timestep}")

        if show and ax is None:  # Only show if it's not part of a mosaic
            plt.show()


    def plot_polar_cartesian_with_z_still(self, timestep, r_max, r_min=0, ax=None, show=True, z_min=None, z_max=None, elevation=5, azimuth=90):
        """Generates a single frame of the 3D polar to Cartesian plot at a given timestep."""

        # Create new figure and axis if ax is not provided
        if ax is None:
            figure = plt.figure(figsize=(10, 7))
            ax = figure.add_subplot(111, projection='3d')
        else:
            figure = plt.gcf()  # Get the current figure if ax is provided

        # Set the view angle
        ax.view_init(elev=elevation, azim=azimuth)

        # Set axis limits
        ax.set_xlim(r_min, r_max)
        ax.set_ylim(0, 2 * np.pi)  # Set theta range to [0, 2*pi]
        
        # If z_min/z_max are provided, set z-limits
        if z_min is not None and z_max is not None:
            ax.set_zlim(z_min, z_max)
        else:
            ax.set_zlim(-r_max, r_max)  # Default range for z axis is [-r_max, r_max]
        
        # Label the axes
        ax.set_xlabel('Radial Distance (r) [m]')
        ax.set_ylabel('Angle (theta) [radians]')
        ax.set_zlabel('Height (z) [m]')

        # Function to convert cartesian coordinates to polar coordinates
        def cartesian_to_polar(x, y):
            r = np.sqrt(x ** 2 + y ** 2)
            theta = np.arctan2(y, x)
            return r, np.mod(theta, 2 * np.pi)

        # Create a list to store legend handles
        legend_handles = []

        # Loop through the moons and plot them
        for body in self.dataset.moons:
            r, theta = cartesian_to_polar(body.pos[timestep, 0], body.pos[timestep, 1])
            z = body.pos[timestep, 2]
            marker = ax.scatter(r, theta, z, color=body.color, label=body.name)
            legend_handles.append(marker)

        # Plot test particles
        r_test, theta_test = cartesian_to_polar(
            self.dataset.relative_positions[timestep, self.dataset.num_moons:, 0],
            self.dataset.relative_positions[timestep, self.dataset.num_moons:, 1]
        )
        z_test = self.dataset.relative_positions[timestep, self.dataset.num_moons:, 2]
        test_marker = ax.scatter(r_test, theta_test, z_test, color="navy", s=.1, label="Test Particles")
        legend_handles.append(test_marker)

        # Plot Saturn (barycenter)
        saturn_marker = ax.scatter(0, 0, 0, color="yellow", marker="x", s=100, label="Saturn (Barycenter)")
        legend_handles.append(saturn_marker)

        # Move legend outside the plot
        figure.legend(handles=legend_handles, loc='right', bbox_to_anchor=(1, 0.5), title="Legend")
        
        # Title of the plot
        plt.title(f"3D Polar Plot at timestep {timestep}")

        # Display the plot if show is True
        if show:
            plt.show()

    def plot_mosaic(self, plot_function, timesteps, r_max, r_min=0, **kwargs):
        """
        Creates a 2x2 mosaic of plots for the given timesteps using the specified plot function.

        Parameters:
            plot_function (function): The function used for plotting (2D or 3D).
            timesteps (list): A list of exactly 4 timesteps to plot.
            r_max (float): Maximum radial distance for the plots.
            r_min (float): Minimum radial distance for the plots.
            **kwargs: Additional keyword arguments to pass to the plot function.
        """
        if len(timesteps) != 4:
            raise ValueError("The timesteps list must contain exactly 4 timesteps.")

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        fig.suptitle("Mosaic Plot of Timesteps", fontsize=16)

        for ax, timestep in zip(axes.flat, timesteps):
            plot_function(timestep, r_max, r_min, ax=ax, show=False, **kwargs)  # Pass the subplot (ax)

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit title
        plt.show()

    def plot_mosaic_3d(self, plot_function, timesteps, r_max, r_min=0, **kwargs):
        """
        Creates a 2x2 mosaic of plots for the given timesteps using the specified plot function.

        Parameters:
            plot_function (function): The function used for plotting (2D or 3D).
            timesteps (list): A list of exactly 4 timesteps to plot.
            r_max (float): Maximum radial distance for the plots.
            r_min (float): Minimum radial distance for the plots.
            **kwargs: Additional keyword arguments to pass to the plot function.
        """
        if len(timesteps) != 4:
            raise ValueError("The timesteps list must contain exactly 4 timesteps.")

        fig = plt.figure(figsize=(12, 10))

        # Create 2x2 subplots with 3D axes
        axes = [fig.add_subplot(2, 2, i + 1, projection='3d') for i in range(4)]

        fig.suptitle("Mosaic Plot of Timesteps", fontsize=16)

        # Loop over axes and timesteps and call the plot function
        for ax, timestep in zip(axes, timesteps):
            plot_function(timestep, r_max, r_min, ax=ax, show=False, **kwargs)  # Pass the 3D subplot (ax)

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit title
        plt.show()