import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np

class AnimationManager:
    def __init__(self, dataset):
        self.dataset = dataset

    def plot_2d(self, coords=[0, 1], n_farthest_filter=10, big_traillength=100, small_traillength=3, frame_time=10,interval=1):
        """
        Animates the simulation, displaying the positions and trails of celestial bodies.

        Args:
            coords (list): Coordinate indices to plot (e.g., [0, 1] for X-Y plane).
            n_farthest_filter (int): Number of moons to exclude from zoomed view based on distance.
            big_traillength (int): Length of the main trail for moons.
            small_traillength (int): Length of the zoomed trail for moons.
            interval (int): Time interval between animation frames.
        """
        # Create a figure with two subplots
        figure, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Set axis limits for the full view
        xmax, xmin = np.max([np.max(body.pos[:, coords[0]]) for body in self.dataset.moons]), np.min([np.min(body.pos[:, coords[0]]) for body in self.dataset.moons])
        ymax, ymin = np.max([np.max(body.pos[:, coords[1]]) for body in self.dataset.moons]), np.min([np.min(body.pos[:, coords[1]]) for body in self.dataset.moons])
        ax.set_xlim(xmin - 0.1 * abs(xmin), xmax + 0.1 * abs(xmax))
        ax.set_ylim(ymin - 0.1 * abs(ymin), ymax + 0.1 * abs(ymax))
        ax.set_aspect('equal', adjustable='box')

        # Filter moons for the zoomed-in view
        nth_largest_indices = np.argsort([np.min(body.pos[:, 0]**2 + body.pos[:, 1]**2) for body in self.dataset.moons])[:-n_farthest_filter]
        filtered_moons = [self.dataset.moons[i] for i in nth_largest_indices]
        ax2.set_xlim(-3e8, 3e8)
        ax2.set_ylim(-3e8, 3e8)
        ax2.set_aspect('equal', adjustable='box')

        # Initialize plot objects for moons and test particles
        moon_lines = {body: ax.plot([], [], label=f"{body.name}", color=body.color, linestyle='-')[0] for body in self.dataset.moons}
        moon_markers = {}
        zoom_lines = {}
        zoom_markers = {}

        # Special marker size for Saturn
        large_marker_size = 15
        default_marker_size = 6

        for body in self.dataset.moons:
            marker_size = large_marker_size if body.name.lower() == "saturn" else default_marker_size
            moon_markers[body] = ax.plot([], [], label=f"{body.name}", color=body.color, marker='o', linestyle='', markersize=marker_size)[0]
            if body in filtered_moons:
                zoom_lines[body] = ax2.plot([], [], label=f"{body.name}", color=body.color, linestyle='-')[0]
                zoom_markers[body] = ax2.plot([], [], label=f"{body.name}", color=body.color, marker='o', linestyle='', markersize=marker_size)[0]

        test_particle_line = ax2.plot([], [], ".", label="Test Particles", color="navy", markersize=1)[0]

        # Initialization function for the animation
        def init():
            for line in moon_lines.values():
                line.set_data([], [])
            for marker in moon_markers.values():
                marker.set_data([], [])
            for zoom_line in zoom_lines.values():
                zoom_line.set_data([], [])
            for zoom_marker in zoom_markers.values():
                zoom_marker.set_data([], [])
            test_particle_line.set_data([], [])
            return list(moon_lines.values()) + list(moon_markers.values()) + list(zoom_lines.values()) + list(zoom_markers.values()) + [test_particle_line]

        # Update function for each frame
        def update(i):
            for body in self.dataset.moons:
                moon_lines[body].set_data(
                    body.pos[max(i - big_traillength, 0):i + 1, coords[0]],  # Include current position
                    body.pos[max(i - big_traillength, 0):i + 1, coords[1]]
                )
                moon_markers[body].set_data(
                    body.pos[i, coords[0]],
                    body.pos[i, coords[1]]
                )
            for body in filtered_moons:
                zoom_lines[body].set_data(
                    body.pos[max(i - small_traillength, 0):i + 1, coords[0]],  # Include current position
                    body.pos[max(i - small_traillength, 0):i + 1, coords[1]]
                )
                zoom_markers[body].set_data(
                    body.pos[i, coords[0]],
                    body.pos[i, coords[1]]
                )
            test_particle_line.set_data(
                self.dataset.relative_positions[i, self.dataset.num_moons:, coords[0]],
                self.dataset.relative_positions[i, self.dataset.num_moons:, coords[1]]
            )
            return list(moon_lines.values()) + list(moon_markers.values()) + \
                list(zoom_lines.values()) + list(zoom_markers.values()) + [test_particle_line]


        # Create the animation and assign it to an attribute to prevent garbage collection
        self.animation = anim.FuncAnimation(
            figure,
            update,
            init_func=init,
            frames=np.arange(0, self.dataset.relative_positions.shape[0], interval),
            interval=frame_time,
            blit=False
        )

        # Set axis labels
        labels = {0: 'X (m)', 1: 'Y (m)', 2: 'Z (m)'}
        ax.set_xlabel(labels[coords[0]])
        ax.set_ylabel(labels[coords[1]])
        ax2.set_xlabel(labels[coords[0]])
        ax2.set_ylabel(labels[coords[1]])

        # Add a legend
        marker_handles = [moon_markers[body] for body in self.dataset.moons]
        figure.legend(handles=marker_handles, loc='center right', bbox_to_anchor=(0.98, 0.5), title="Moons")

        # Adjust layout for better readability
        plt.subplots_adjust(left=0.05, right=0.85, top=0.95, bottom=0.05, wspace=0.15)

        plt.show()


    def plot_3d(self, n_farthest_filter=10, big_traillength=100, small_traillength=3, frame_time=100, interval=1):
        figure = plt.figure(figsize=(14, 7))

        # Create two 3D subplots
        ax = figure.add_subplot(121, projection='3d')
        ax2 = figure.add_subplot(122, projection='3d')

        # Set wide view axis limits
        xmax, xmin = np.max([np.max(body.pos[:, 0]) for body in self.dataset.moons]), np.min([np.min(body.pos[:, 0]) for body in self.dataset.moons])
        ymax, ymin = np.max([np.max(body.pos[:, 1]) for body in self.dataset.moons]), np.min([np.min(body.pos[:, 1]) for body in self.dataset.moons])
        zmax, zmin = np.max([np.max(body.pos[:, 2]) for body in self.dataset.moons]), np.min([np.min(body.pos[:, 2]) for body in self.dataset.moons])

        # Add debugging prints for axis limits
        # print(f"X Range: {xmin} to {xmax}")
        # print(f"Y Range: {ymin} to {ymax}")
        # print(f"Z Range: {zmin} to {zmax}")

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(zmin, zmax)

        # Zoomed-in view filter based on distance
        nth_largest_indices = np.argsort([np.min(body.pos[:, 0]**2 + body.pos[:, 1]**2 + body.pos[:, 2]**2) for body in self.dataset.moons])[:-n_farthest_filter]
        filtered_moons = [self.dataset.moons[i] for i in nth_largest_indices]

        # Zoomed-in axis limits
        zoom_range = 3e8
        ax2.set_xlim(-zoom_range, zoom_range)
        ax2.set_ylim(-zoom_range, zoom_range)
        ax2.set_zlim(-zoom_range, zoom_range)

        # Initialize moon plot objects
        moon_lines = {body: ax.plot([], [], [], label=f"{body.name}", color=body.color, linestyle='-')[0] for body in self.dataset.moons}
        moon_markers = {body: ax.plot([], [], [], label=f"{body.name}", color=body.color, marker='o', linestyle='')[0] for body in self.dataset.moons}

        zoom_lines = {body: ax2.plot([], [], [], label=f"{body.name}", color=body.color, linestyle='-')[0] for body in filtered_moons}
        zoom_markers = {body: ax2.plot([], [], [], label=f"{body.name}", color=body.color, marker='o', linestyle='')[0] for body in filtered_moons}

        # Initialize test particle plot objects
        test_particle_line = ax2.plot([], [], [], ".", label="Test Particles", color="navy", markersize=1)[0]

        def init():
            # Initialize moons
            for line in moon_lines.values():
                line.set_data_3d([], [], [])
            for marker in moon_markers.values():
                marker.set_data_3d([], [], [])
            for line in zoom_lines.values():
                line.set_data_3d([], [], [])
            for marker in zoom_markers.values():
                marker.set_data_3d([], [], [])

            test_particle_line.set_data_3d([], [], [])

            # Add debugging prints for init function
            #print("Init function called.")

            return list(moon_lines.values()) + list(moon_markers.values()) + \
                list(zoom_lines.values()) + list(zoom_markers.values()) + \
                [test_particle_line]

        def update(i):
            # Update moons in wide view
            for body in self.dataset.moons:
                moon_lines[body].set_data_3d(
                    body.pos[max(i - big_traillength, 0):i + 1, 0],  # Include current position
                    body.pos[max(i - big_traillength, 0):i + 1, 1],
                    body.pos[max(i - big_traillength, 0):i + 1, 2]
                )
                moon_markers[body].set_data_3d(
                    body.pos[i, 0],
                    body.pos[i, 1],
                    body.pos[i, 2]
                )

            # Update moons in zoomed-in view
            for body in filtered_moons:
                zoom_lines[body].set_data_3d(
                    body.pos[max(i - small_traillength, 0):i + 1, 0],  # Include current position
                    body.pos[max(i - small_traillength, 0):i + 1, 1],
                    body.pos[max(i - small_traillength, 0):i + 1, 2]
                )
                zoom_markers[body].set_data_3d(
                    body.pos[i, 0],
                    body.pos[i, 1],
                    body.pos[i, 2]
                )

            # Update test particles
            test_particle_line.set_data_3d(
                self.dataset.relative_positions[i, self.dataset.num_moons:, 0],
                self.dataset.relative_positions[i, self.dataset.num_moons:, 1],
                self.dataset.relative_positions[i, self.dataset.num_moons:, 2]
            )

            return list(moon_lines.values()) + list(moon_markers.values()) + \
                list(zoom_lines.values()) + list(zoom_markers.values()) + \
                [test_particle_line]


        # Create animation
        self.animation = anim.FuncAnimation(
            figure,
            update,
            init_func=init,
            frames=np.arange(0, self.dataset.moons[0].pos.shape[0], interval),
            interval=frame_time,
            blit=False
        )

        # Set labels and legend
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_zlabel('Z (m)')

        handles = [moon_markers[body] for body in self.dataset.moons]
        figure.legend(handles=handles, loc='center right', bbox_to_anchor=(0.98, 0.5), title="Moons")
        plt.subplots_adjust(left=0.05, right=0.85, top=0.95, bottom=0.05, wspace=0.2)

        plt.show()


    def plot_centered(self, moon_name, width, frame_time=10, trail_length=100, target_trail_length=150, interval=1):
        """Single-axis plot centered on a specific moon with no trails for test particles"""
        figure, ax = plt.subplots(figsize=(14, 7))
        target_moon = next((body for body in self.dataset.moons if body.name.lower() == moon_name.lower()), None)
        if not target_moon:
            raise ValueError(f"Moon '{moon_name}' not found!")

        ax.set_xlim(-width / 2, width / 2)
        ax.set_ylim(-width / 2, width / 2)
        ax.set_aspect('equal', adjustable='box')

        # Create a larger marker for the target moon (to make it more visible)
        moon_marker = ax.plot([], [], label=f"{target_moon.name}", color=target_moon.color, marker='o', markersize=8, linestyle='', zorder=5)[0]
        
        # Initialize trail for target moon
        target_trail, = ax.plot([], [], color=target_moon.color, linestyle='-', alpha=0.5, zorder=2)

        # Create markers and trails for other moons
        other_markers = {
            body: ax.plot([], [], label=f"{body.name}", color=body.color, marker='o', markersize=6, linestyle='', zorder=3)[0]
            for body in self.dataset.moons if body != target_moon
        }
        
        other_trails = {
            body: ax.plot([], [], color=body.color, linestyle='-', alpha=0.5, zorder=1)[0]
            for body in self.dataset.moons if body != target_moon
        }

        # Test particle marker without trails
        test_particle_marker = ax.plot([], [], ".", label="Test Particles", color="navy", markersize=1, zorder=0)[0]

        # Initialize the trail for the centered moon (target_moon)
        target_moon_trail = ax.plot([], [], color='red', linestyle='--', alpha=0.5, zorder=2)[0]

        def init():
            moon_marker.set_data([], [])
            target_trail.set_data([], [])
            target_moon_trail.set_data([], [])
            for marker in other_markers.values():
                marker.set_data([], [])
            for trail in other_trails.values():
                trail.set_data([], [])
            test_particle_marker.set_data([], [])
            return [moon_marker] + list(other_markers.values()) + [test_particle_marker] + list(other_trails.values()) + [target_trail, target_moon_trail]

        def update(i):
            # Update position for target moon marker
            center_x, center_y = target_moon.pos[i, 0], target_moon.pos[i, 1]
            ax.set_xlim(center_x - width / 2, center_x + width / 2)
            ax.set_ylim(center_y - width / 2, center_y + width / 2)
            moon_marker.set_data(center_x, center_y)

            # Update trail for the target moon
            trail_x = target_moon.pos[max(0, i - trail_length):i + 1, 0]
            trail_y = target_moon.pos[max(0, i - trail_length):i + 1, 1]
            target_trail.set_data(trail_x, trail_y)

            # Update trail for the centered target moon
            target_moon_trail_x = target_moon.pos[max(0, i - target_trail_length):i + 1, 0]
            target_moon_trail_y = target_moon.pos[max(0, i - target_trail_length):i + 1, 1]
            target_moon_trail.set_data(target_moon_trail_x, target_moon_trail_y)

            # Update positions and trails for the other moons
            for body, marker in other_markers.items():
                marker.set_data(body.pos[i, 0], body.pos[i, 1])
                trail_x = body.pos[max(0, i - trail_length):i + 1, 0]
                trail_y = body.pos[max(0, i - trail_length):i + 1, 1]
                other_trails[body].set_data(trail_x, trail_y)

            # Update test particles (positions only, no trails)
            test_particle_marker.set_data(
                self.dataset.relative_positions[i, self.dataset.num_moons:, 0],
                self.dataset.relative_positions[i, self.dataset.num_moons:, 1]
            )

            return [moon_marker] + list(other_markers.values()) + [test_particle_marker] + list(other_trails.values()) + [target_trail, target_moon_trail]

        self.animation = anim.FuncAnimation(
            figure,
            update,
            init_func=init,
            frames=np.arange(0, self.dataset.relative_positions.shape[0], interval),
            interval=frame_time,
            blit=False
        )

        # Set labels and legend
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')

        # Add a customized legend with the markers outside the plot
        marker_handles = [moon_marker] + list(other_markers.values()) + [test_particle_marker]
        figure.legend(handles=marker_handles, loc='center left', bbox_to_anchor=(0.7, 0.5), title="Bodies")

        plt.subplots_adjust(right=0.85)  # Adjust the plot area to make space for the legend

        plt.show()

    def plot_centered_3d(self, moon_name, width, frame_time=10, trail_length=100, target_trail_length=150, interval=1):
        """3D plot centered on a specific moon with no trails for test particles"""
        figure = plt.figure(figsize=(14, 7))
        ax = figure.add_subplot(111, projection='3d')

        # Find the target moon
        target_moon = next((body for body in self.dataset.moons if body.name.lower() == moon_name.lower()), None)
        if not target_moon:
            raise ValueError(f"Moon '{moon_name}' not found!")

        # Set initial limits for the 3D view
        ax.set_xlim(-width / 2, width / 2)
        ax.set_ylim(-width / 2, width / 2)
        ax.set_zlim(-width / 2, width / 2)

        # Create a larger marker for the target moon
        moon_marker = ax.plot([], [], [], label=f"{target_moon.name}", color=target_moon.color, marker='o', markersize=8, linestyle='', zorder=5)[0]

        # Initialize trail for the target moon
        target_trail, = ax.plot([], [], [], color=target_moon.color, linestyle='-', alpha=0.5, zorder=2)

        # Create markers and trails for other moons
        other_markers = {
            body: ax.plot([], [], [], label=f"{body.name}", color=body.color, marker='o', markersize=6, linestyle='', zorder=3)[0]
            for body in self.dataset.moons if body != target_moon
        }

        other_trails = {
            body: ax.plot([], [], [], color=body.color, linestyle='-', alpha=0.5, zorder=1)[0]
            for body in self.dataset.moons if body != target_moon
        }

        # Test particle marker without trails
        test_particle_marker = ax.plot([], [], [], ".", label="Test Particles", color="navy", markersize=1, zorder=0)[0]

        def init():
            moon_marker.set_data_3d([], [], [])
            target_trail.set_data_3d([], [], [])
            for marker in other_markers.values():
                marker.set_data_3d([], [], [])
            for trail in other_trails.values():
                trail.set_data_3d([], [], [])
            test_particle_marker.set_data_3d([], [], [])
            return [moon_marker] + list(other_markers.values()) + [test_particle_marker] + list(other_trails.values()) + [target_trail]

        def update(i):
            # Center the view on the target moon
            center_x, center_y, center_z = target_moon.pos[i, 0], target_moon.pos[i, 1], target_moon.pos[i, 2]
            ax.set_xlim(center_x - width / 2, center_x + width / 2)
            ax.set_ylim(center_y - width / 2, center_y + width / 2)
            ax.set_zlim(center_z - width / 2, center_z + width / 2)

            # Update position for target moon marker
            moon_marker.set_data_3d(center_x, center_y, center_z)

            # Update trail for the target moon
            trail_x = target_moon.pos[max(0, i - trail_length):i + 1, 0]
            trail_y = target_moon.pos[max(0, i - trail_length):i + 1, 1]
            trail_z = target_moon.pos[max(0, i - trail_length):i + 1, 2]
            target_trail.set_data_3d(trail_x, trail_y, trail_z)

            # Update positions and trails for the other moons
            for body, marker in other_markers.items():
                marker.set_data_3d(body.pos[i, 0], body.pos[i, 1], body.pos[i, 2])
                trail_x = body.pos[max(0, i - trail_length):i + 1, 0]
                trail_y = body.pos[max(0, i - trail_length):i + 1, 1]
                trail_z = body.pos[max(0, i - trail_length):i + 1, 2]
                other_trails[body].set_data_3d(trail_x, trail_y, trail_z)

            # Update test particles (positions only, no trails)
            test_particle_marker.set_data_3d(
                self.dataset.relative_positions[i, self.dataset.num_moons:, 0],
                self.dataset.relative_positions[i, self.dataset.num_moons:, 1],
                self.dataset.relative_positions[i, self.dataset.num_moons:, 2]
            )

            return [moon_marker] + list(other_markers.values()) + [test_particle_marker] + list(other_trails.values()) + [target_trail]

        self.animation = anim.FuncAnimation(
            figure,
            update,
            init_func=init,
            frames=np.arange(0, self.dataset.relative_positions.shape[0], interval),
            interval=frame_time,
            blit=False
        )

        # Set labels and legend
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')

        # Add a customized legend with the markers outside the plot
        marker_handles = [moon_marker] + list(other_markers.values()) + [test_particle_marker]
        figure.legend(handles=marker_handles, loc='center left', bbox_to_anchor=(0.75, 0.5), title="Bodies")

        plt.subplots_adjust(right=0.85)  # Adjust the plot area to make space for the legend

        plt.show()

    def plot_polar_cartesian(self, r_max, r_min=0,theta_max = 2*np.pi ,frame_time=10, trail_length=100, interval=1, heatmap=False):
        """Polar to Cartesian plot with (r, theta) coordinates for all moons, Saturn, and test particles, with an optional heatmap for test particles."""
        figure, ax = plt.subplots(figsize=(14, 7))
        
        # Set up the axis: r is on the x-axis, theta (0 to 2pi) is on the y-axis
        ax.set_xlim(r_min, r_max)
        ax.set_ylim(0, theta_max)
        ax.set_xlabel('Radial Distance (r) [m]')
        ax.set_ylabel('Angle (theta) [radians]')
        
        # Helper function to convert Cartesian (x, y) to polar (r, theta)
        def cartesian_to_polar(x, y):
            r = np.sqrt(x ** 2 + y ** 2)
            theta = np.arctan2(y, x)
            return r, np.mod(theta, 2 * np.pi)  # Ensure theta is in the range [0, 2pi]

        # Create markers and trails for all moons, including Saturn at the origin
        moon_markers = {
            body: ax.plot([], [], label=f"{body.name}", color=body.color, marker='o', markersize=6, linestyle='')[0]
            for body in self.dataset.moons
        }
        
        moon_trails = {
            body: ax.plot([], [], color=body.color, linestyle='-', alpha=0.5)[0]
            for body in self.dataset.moons
        }

        # Test particles (either as individual markers or as a heatmap)
        if not heatmap:
            test_particle_marker = ax.plot([], [], ".", label="Test Particles", color="navy", markersize=1)[0]
        else:
            # Create an empty 2D histogram for the heatmap (r and theta grid)
            heatmap_data, xedges, yedges = np.histogram2d([], [], bins=(100, 100), range=[[r_min, r_max], [0, 2 * np.pi]])
            heatmap_img = ax.imshow(heatmap_data.T, extent=[r_min, r_max, 0, 2 * np.pi], origin='lower', aspect='auto', cmap='viridis', alpha=0.8)

        # Saturn at origin (barycenter) with an "X" marker
        saturn_marker = ax.plot([0], [0], "x", label="Saturn (Barycenter)", color="yellow", markersize=10)[0]

        def init():
            # Initialize markers and trails to empty
            for marker in moon_markers.values():
                marker.set_data([], [])
            for trail in moon_trails.values():
                trail.set_data([], [])
            if not heatmap:
                test_particle_marker.set_data([], [])
                return list(moon_markers.values()) + [test_particle_marker] + list(moon_trails.values()) + [saturn_marker]
            else:
                heatmap_img.set_data(np.zeros_like(heatmap_img.get_array()))
                return list(moon_markers.values()) + list(moon_trails.values()) + [heatmap_img, saturn_marker]

        def update(i):
            # Update the positions and trails for all moons in polar coordinates
            for body, marker in moon_markers.items():
                r, theta = cartesian_to_polar(body.pos[i, 0], body.pos[i, 1])
                marker.set_data(r, theta)
                
                # Update the trail of the moons in polar coordinates
                trail_r, trail_theta = cartesian_to_polar(
                    body.pos[max(0, i - trail_length):i + 1, 0],
                    body.pos[max(0, i - trail_length):i + 1, 1]
                )
                moon_trails[body].set_data(trail_r, trail_theta)

            # Plot title:
            plt.title(f"frame {i}")
            
            if not heatmap:
                # Update the test particles in polar coordinates
                r_test, theta_test = cartesian_to_polar(
                    self.dataset.relative_positions[i, self.dataset.num_moons:, 0],
                    self.dataset.relative_positions[i, self.dataset.num_moons:, 1]
                )
                test_particle_marker.set_data(r_test, theta_test)

                return list(moon_markers.values()) + [test_particle_marker] + list(moon_trails.values()) + [saturn_marker]
            
            else:
                # Update the heatmap for test particles
                r_test, theta_test = cartesian_to_polar(
                    self.dataset.relative_positions[i, self.dataset.num_moons:, 0],
                    self.dataset.relative_positions[i, self.dataset.num_moons:, 1]
                )
                
                # Recreate the 2D histogram with current test particle positions
                heatmap_data, xedges, yedges = np.histogram2d(r_test, theta_test, bins=(100, 100), range=[[r_min, r_max], [0, 2 * np.pi]])
                
                # Update the heatmap image data
                heatmap_img.set_data(heatmap_data.T)
                heatmap_img.set_clim(vmin=0, vmax=np.max(heatmap_data))  # Adjust color limits to avoid full saturation

                return list(moon_markers.values()) + list(moon_trails.values()) + [heatmap_img, saturn_marker]

        # Create the animation
        self.animation = anim.FuncAnimation(
            figure,
            update,
            init_func=init,
            frames=np.arange(0, self.dataset.relative_positions.shape[0], interval),
            interval=frame_time,
            blit=False
        )

        # Add a customized legend with the markers outside the plot
        if not heatmap:
            marker_handles = list(moon_markers.values()) + [test_particle_marker, saturn_marker]
        else:
            marker_handles = list(moon_markers.values()) + [saturn_marker]

        figure.legend(handles=marker_handles, loc='center left', bbox_to_anchor=(0.85, 0.5), title="Bodies")

        plt.subplots_adjust(right=0.85)  # Adjust the plot area to make space for the legend

        plt.show()
    
    def plot_polar_cartesian_with_z(self, r_max, r_min=0, z_min=None, z_max=None,elevation = 5 , azimuth = 90 ,frame_time=10, trail_length=100, interval=1, heatmap=False):
        """3D Polar to Cartesian plot with (r, theta, z) coordinates for all moons, Saturn, and test particles."""
        figure = plt.figure(figsize=(14, 7))
        ax = figure.add_subplot(111, projection='3d')  # Create a 3D subplot
        
        # Set the viewing angle (elevation and azimuth)
        ax.view_init(elev=elevation, azim=azimuth) 
    
        # Set up the axis: r is on the x-axis, theta (0 to 3pi) is on the y-axis, and z is on the z-axis
        ax.set_xlim(r_min, r_max)
        ax.set_ylim(-2*np.pi, 2 * np.pi)  # Extend theta axis from -2pi to 2pi
        if z_min is not None and z_max is not None:
            ax.set_zlim(z_min, z_max)  # Set z limits based on user input
        else:
            ax.set_zlim(-r_max, r_max)  # Default z limits based on r_max
        ax.set_xlabel('Radial Distance (r) [m]')
        ax.set_ylabel('Angle (theta) [radians]')
        ax.set_zlabel('Height (z) [m]')
        
        # Helper function to convert Cartesian (x, y) to polar (r, theta)
        def cartesian_to_polar(x, y):
            r = np.sqrt(x ** 2 + y ** 2)
            theta = np.arctan2(y, x)
            return r, np.mod(theta, 2 * np.pi)  # Ensure theta is in the range [0, 2pi]

        # Create markers and trails for all moons, including Saturn at the origin
        moon_markers = {
            body: ax.plot([], [], [], label=f"{body.name}", color=body.color, marker='o', markersize=6, linestyle='')[0]
            for body in self.dataset.moons
        }
        
        moon_trails = {
            body: ax.plot([], [], [], color=body.color, linestyle='-', alpha=0.5)[0]
            for body in self.dataset.moons
        }

        # Test particles (either as individual markers or as a heatmap)
        if not heatmap:
            test_particle_marker = ax.plot([], [], [], ".", label="Test Particles", color="navy", markersize=0.5)[0]
        else:
            # We cannot use a 3D heatmap easily, so let's leave this part out for simplicity
            pass

        # Saturn at origin (barycenter) with an "X" marker
        saturn_marker = ax.plot([0], [0], [0], "x", label="Saturn (Barycenter)", color="yellow", markersize=10)[0]

        def init():
            # Initialize markers and trails to empty
            for marker in moon_markers.values():
                marker.set_data_3d([], [], [])
            for trail in moon_trails.values():
                trail.set_data_3d([], [], [])
            if not heatmap:
                test_particle_marker.set_data_3d([], [], [])
                return list(moon_markers.values()) + [test_particle_marker] + list(moon_trails.values()) + [saturn_marker]
            else:
                return list(moon_markers.values()) + list(moon_trails.values()) + [saturn_marker]

        def update(i):
            # Update the positions and trails for all moons in polar coordinates
            for body, marker in moon_markers.items():
                r, theta = cartesian_to_polar(body.pos[i, 0], body.pos[i, 1])
                z = body.pos[i, 2]  # Take the z-direction (height) from the third dimension

                # Plot only points within the theta range [0, 2pi]
                if theta <= 2 * np.pi:
                    marker.set_data_3d([r], [theta], [z])
                
                # Update the trail of the moons in polar coordinates
                trail_r, trail_theta = cartesian_to_polar(
                    body.pos[max(0, i - trail_length):i + 1, 0],
                    body.pos[max(0, i - trail_length):i + 1, 1]
                )
                trail_z = body.pos[max(0, i - trail_length):i + 1, 2]

                # Plot only trails within the theta range [0, 2pi]
                valid_idx = trail_theta <= 2 * np.pi
                moon_trails[body].set_data_3d(trail_r[valid_idx], trail_theta[valid_idx], trail_z[valid_idx])

            # Plot title:
            plt.title(f"frame {i}")
            
            if not heatmap:
                # Update the test particles in polar coordinates
                r_test, theta_test = cartesian_to_polar(
                    self.dataset.relative_positions[i, self.dataset.num_moons:, 0],
                    self.dataset.relative_positions[i, self.dataset.num_moons:, 1]
                )
                z_test = self.dataset.relative_positions[i, self.dataset.num_moons:, 2]  # Test particles in z-direction

                # Plot only points within the theta range [0, 2pi]
                valid_test_idx = theta_test <= 2 * np.pi
                test_particle_marker.set_data_3d(r_test[valid_test_idx], theta_test[valid_test_idx], z_test[valid_test_idx])

                return list(moon_markers.values()) + [test_particle_marker] + list(moon_trails.values()) + [saturn_marker]
            else:
                return list(moon_markers.values()) + list(moon_trails.values()) + [saturn_marker]

        # Create the animation
        self.animation = anim.FuncAnimation(
            figure,
            update,
            init_func=init,
            frames=np.arange(0, self.dataset.relative_positions.shape[0], interval),
            interval=frame_time,
            blit=False
        )

        # Add a customized legend with the markers outside the plot
        if not heatmap:
            marker_handles = list(moon_markers.values()) + [test_particle_marker, saturn_marker]
        else:
            marker_handles = list(moon_markers.values()) + [saturn_marker]

        figure.legend(handles=marker_handles, loc='center left', bbox_to_anchor=(0.85, 0.5), title="Bodies")

        plt.subplots_adjust(right=0.85)  # Adjust the plot area to make space for the legend

        plt.show()
