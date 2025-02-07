import numpy as np
import matplotlib.pyplot as plt
import SaturnMoonLibrary.SaturnMoonSimulation as sms
import os

class EnergyAnalysis:
    def __init__(self, dataset):
        self.dataset = dataset
    def _get_masses_moons(self):
        folder_name=self.dataset.header["Initial Data Folder"]
        local_path_to_horizons =  os.path.join(".", "SaturnModelDatabase","horizons_data")
        file_path = os.path.join(local_path_to_horizons, folder_name, "rotated_dictionary_" + folder_name + ".txt")
        return {moon: info["Mass"] for moon,info in sms.read_dictionary_from_file(file_path).items()}
    def energy(self):
        """calculates the total energy of the system, neglecting masses smaller than moons and oblateness of saturn"""
        G = 6.674 * 10**-11  # m^3 kg^-1 s^-2
        masses= self._get_masses_moons()
        M=masses["Saturn"]
        J2=self.dataset.header["J2"]
        Req=6.3781e6#equatorial radius saturn
        potential=np.zeros(len(self.dataset.positions))
        for j,p1 in enumerate(self.dataset.moons):
            for p2 in self.dataset.moons[j+1:20]:
                if p1==p2: continue
                potential+=-2*G*masses[p1.name]*masses[p2.name]/np.sqrt(np.sum((p1.pos-p2.pos)**2,axis=1))
            if j==0: continue
            dist=p1.pos-self.dataset.moons[0].pos
            r=np.sqrt(dist[:,0]**2+dist[:,1]**2+dist[:,2]**2)
            potential+=G*(M+masses[p1.name])/r*J2*(Req/r)**2 *1/2*(3*(dist[:,2]/r)**2-1)
        kinetic=np.sum([masses[it.name]*np.sum(self.dataset.positions[:,index,3:]**2,axis=1) for index,it in enumerate(self.dataset.moons)],axis=0)
        return kinetic+potential
    def centre_of_mass(self):
        """calculates position and velocity of centre of mass of the system, while neglecting mass of anything smaller than a moon
        returns centre of mass and average velocity
        """
        masses= self._get_masses_moons()
        totx=np.sum([masses[moon.name]*self.dataset.positions[:,index,:3] for index,moon in enumerate(self.dataset.moons)],axis=0)/np.sum([masses[moon.name] for moon in self.dataset.moons])
        totv=np.sum([masses[moon.name]*self.dataset.positions[:,index,3:] for index,moon in enumerate(self.dataset.moons)],axis=0)/np.sum([masses[moon.name] for moon in self.dataset.moons])
        return totx,totv
    def plot_energy(self,scale="absolute",color=None,Label=None):
        """plots the total energy of the system, in absolute scale or relative to the starting value
        inputs: scale 'absolute' or 'relative'
        """
        Energy=self.energy()
        dt=self.dataset.header["dt"]*(self.dataset.header["Saved Points Modularity"])
        if scale=="absolute":
            plot=plt.plot(dt/60/60/24*np.arange(0,len(Energy),1),Energy,color=color,label=Label)
            plt.ylabel("total energy (J)")
        elif scale=="relative":
            plot=plt.plot(dt/60/60/24*np.arange(0,len(Energy),1),(Energy/Energy[0]-1)*1e6,color=color,label=Label)
            plt.ylabel("relative error (ppm)")
        else: raise NameError("Use either 'absolute' or 'relative' for the scale parameter")
        plt.xlabel("time (days)")
        return plot
    def plot_centre(self):
        figur=plt.figure()
        gi = figur.add_gridspec(2, hspace=0)
        (ax1,ax2)=gi.subplots(sharex=True)
        dt=self.dataset.header["dt"]*(self.dataset.header["Saved Points Modularity"])
        ax1.plot(dt/60/60/24*np.arange(0,len(totx),1),totv*60*60*24,label=["x","y","z"])
        ax1.set_ylabel("V (m/d)")
        ax1.legend()
        ax2.plot(dt/60/60/24*np.arange(0,len(totx),1),totx-dt*np.cumsum(np.concatenate((np.zeros((1,3)),totv[:-1]),axis=0),axis=0),label=["x","y","z"])
        ax2.set_ylabel("pos (m)")
        totx,totv=self.centre_of_mass()
        figur.supxlabel("time (days)")
    #local path to horizons,
