import numpy as np

class CelestialBody:
    """
    Represents a celestial body (e.g., moon or test particle) with attributes such as name, color, position, and trail options.

    Attributes:
        name (str): Name of the celestial body.
        color (str): Color used for plotting the body.
        pos (ndarray): 3D positional data (timesteps x 3).
        trail (bool): Whether the body should have a trailing line when animated.
    """
    def __init__(self, name, color, pos,vel=0,trail=False):
        self.name = name
        self.color = color
        self.pos = pos  # Positional data as a 3D NumPy array (timesteps x 3)se
        self.vel = vel
        self.trail = trail
    def calculate_orbital_elements(self):
        """
        Takes a particle and returns its corresponding orbital parameters as values.
        Parameters:
        - moon_data: dictionary with moon indices and names
        Returns:
        - result: dictionary with moon names as keys and orbital parameters as values
        """
        G = 6.674 * 10**-11  # m^3 kg^-1 s^-2
        M = 5.6834 * 10**26  # kg (mass of the primary body, e.g., planet or star)
        # Placeholder for storing orbital elements over time
        orbital_elements = np.zeros((self.pos.shape[0], 6))  # 6 orbital elements for each timestep
        x = self.pos[:, :]
        v = self.vel[:, :] 
        ###### center SHOULD be saturn is not currently

        # Angular momentum vector
        h = np.cross(x, v,axis=1)
        h_abs = np.sqrt(h[:,0]**2+h[:,1]**2+h[:,2]**2)

        # Node vector
        n = np.cross([0, 0, 1], h)
        n_abs = np.sqrt(n[:,0]**2+n[:,1]**2+n[:,1]**2)

        # Eccentricity vector
        e_vec = np.cross(v, h) / (G * M) - np.transpose(np.transpose(x[:,:])/np.sqrt(x[:,0]**2+x[:,1]**2+x[:,2]**2))
        e = np.sqrt(e_vec[:,0]**2+e_vec[:,1]**2+e_vec[:,2]**2)

        # Semi-major axis
        a = h_abs**2 / (G * M*(1-e**2))

        # Inclination
        i = np.arccos(h[:,2] / h_abs)

        # Longitude of ascending node
        Omega_temp=np.arccos(n[:,0]/n_abs) #longitude of ascending node
        if np.all(n[:,1]>=0):
            Omega=Omega_temp
        elif np.all(n[:,1]<0):
            Omega=2*np.pi-Omega_temp
        else:
            Omega=2*np.pi*(n[:,1]<0)+(1-2*(n[:,1]<0))*Omega_temp # This case should only happen in the rare event that n is very dependent on time

        # Argument of periapsis
        omega_temp=np.arccos(np.einsum("ij,ij->i",n,e_vec)/n_abs/e) #einsum works, but most importantly best I can find, quite fast as well can do 1e4 computations in .4s
        if np.all(e_vec[:,2]>=0):
            omega=omega_temp
        elif np.all(e_vec[:,2]<0):
            omega=2*np.pi-omega_temp
        else:
            omega=2*np.pi*(e_vec[:,2]<0)+(1-2*(e_vec[:,2]<0))*omega_temp # This case should only happen in the rare event that n is very dependent on time

        # Orbital period
        T = 2 * np.pi * np.sqrt(a**3 / (G * M))

        # Store orbital elements for this timestep
        orbital_params = {
            "inclination": np.degrees(i),
            "ascending_node_longitude": np.degrees(Omega),
            "argument_of_periapsis": np.degrees(omega),
            "eccentricity": e,
            "semimajor_axis": a,
            "period": T
        }
        vectors = {
            "angular_momentum_vector": h,
            "ascending_node_vector": n,
            "eccentricity_vector": e_vec
        }
        return (orbital_params, vectors)

class Dataset:
    """
    Handles the simulation dataset, including reading the header, extracting positional data,
    and initializing celestial bodies for visualization.

    Attributes:
        filepath (str): Path to the binary simulation data file.
        header (dict): Header information extracted from the file.
        num_moons (int): Number of moons in the dataset.
        num_test_particles (int): Number of test particles in the dataset.
        positions (ndarray): 3D array of positions and velocities for all objects (timesteps x objects x 6).
        moons (list): List of CelestialBody objects representing moons.
        test_particles (list): List of CelestialBody objects representing test particles.
    """
    def __init__(self, filepath):
        self.filepath = filepath
        self.header = self._read_header()
        self.num_moons = int(self.header['Moon Count'])
        self.num_test_particles = int(self.header['Number of Test Particles'])
        self.positions = self._read_binary_file()
        self.moons, self.test_particles = self._initialize_bodies()

    def _read_header(self):
        """
        Reads and parses the header section of the binary file.
 
        Returns:
            dict: Parsed header as a dictionary.
        """
        standard_values={
            'Number of Test Particles': 0,
            'Inner Radius': None,
            'Outer Radius': None,
            'J2': -0.01629,
            'Ring Folder': None,
            'Number of Radial Bins': 0,
            'Number of Azimuthal Bins': 0,
            'Theta Max': 6.283185307179586,
            'Include Shear Forces': 'False',
            'Include Particle-Moon Collisions': None,
            'Initialisation Method': 'standard',
            'Runtime': None}
        critical_values=['Moon Names',
            'Moon Count',
            'Initial Data Folder',
            'Epoch',
            'dt',
            'Timesteps',
            'Saved Points Modularity',
            'Skipped Timesteps',
            'Numerical Integrator']
        header = {}
        with open(self.filepath, 'rb') as file:
            while True:
                line = file.readline()
                if b"End of Header" in line:
                    break
                key, value = line.decode('utf-8').strip().split(':', 1)
                header[key.strip()] = self._convert_value(value.strip())
        keys=[]
        for key in critical_values:
            if key not in header.keys():
                keys.append(key)
        if len(keys):
            raise AttributeError(f"critial values {keys} not in the header")
        keys=[]
        for key,item in standard_values.items():
            if key not in header.keys():
                keys.append(key)
                header[key]=item
        if len(keys):
            print(f"{keys} set to standard value")
        return header

    def _convert_value(self, value):
        """
        Converts header values to their appropriate data types.

        Args:
            value (str): Header value to be converted.

        Returns:
            int, float, or str: Converted value.
        """
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            return value

    def _read_binary_file(self):
        """
        Reads the binary data section of the file, skipping the header.

        Returns:
            ndarray: Reshaped positional and velocity data (timesteps x objects x 6).
        """
        length_header = len(self._read_header_raw()) + len("End of Header\n")
        data = np.fromfile(self.filepath, dtype=np.float64, offset=length_header)
        print(data)
        return data.reshape(-1, self.num_moons + self.num_test_particles, 6)

    def _read_header_raw(self):
        """
        Reads the raw header for calculating its length.

        Returns:
            bytes: Raw header data.
        """
        header = []
        with open(self.filepath, 'rb') as file:
            while True:
                line = file.readline()
                if b"End of Header" in line:
                    break
                header.append(line)
        return b"".join(header)

    def _initialize_bodies(self):
        """
        Initializes celestial bodies (moons and test particles) with positional data.
 
        Returns:
            tuple: A list of moon CelestialBody objects and test particle CelestialBody objects.
        """
        relative_positions = np.transpose(np.transpose(self.positions[:,:,:3],[1,0,2])-self.positions[:,0,:3],[1,0,2])
        self.relative_positions=relative_positions
        relative_velocities=np.transpose(np.transpose(self.positions[:,:,3:],[1,0,2])-self.positions[:,0,3:],[1,0,2])
        self.relative_velocities=relative_velocities
        moon_names = self.header['Moon Names'].split(', ')
        body_colors = ['yellow', 'red', 'chartreuse', 'lightblue', 'orange', 'brown', 'blue', 'pink', 'red', 'black',
                       'green', 'purple', 'cyan', 'magenta', 'gold', 'silver', 'lime', 'navy', 'maroon', 'crimson']
        moon_colors = body_colors[:self.num_moons]
 
        moons = [
            CelestialBody(moon_names[i], moon_colors[i], relative_positions[:,i],vel=relative_velocities[:,i], trail=True)
            for i in range(self.num_moons)
        ]# 0 should be saturn, so position relative to saturn
        # test_particle_positions = self.positions[:, self.num_moons:, :3]-np.stack([self.positions[:,0,:3] for i in range(self.positions.shape[1]-self.num_moons)],axis=1) # 0 should be saturn, so position relative to saturn
        test_particles = [
            CelestialBody(str(i), "navy", relative_positions[:,self.num_moons+i], vel=relative_velocities[:,self.num_moons+i], trail=False)
            for i in range(self.num_test_particles)
        ]
 
        return moons, test_particles
 