Simulating the rings of Saturn to model the waves formed in its rings.
author: Bos, S., Dirkzwager, F.J.M., Enthoven, A., and van Uffelen, K.D.
date: 07-02-2025

Summary:
On this github page all the important code that has been used for this project is located.
This folder has a general structure existing of an Archive Folder which has most of the junk code of previous versions stored,
this is not important for the rest of the code.THen there is the READ Me folder where we are now located that also contains an exact tree of the file structure.
Then there is the SimulationModelDatabase folder that has the four main data folders: horizons_data, horizons_long_format_data ,ring_data, simulation_data.
Here horizons_data contains all initial datasets that are used in the simulations, all generated from the JPL Horizons Database.
Simmilarly in horizons_long_format_data there are datasets of periods of time as reference for our simulation results, alse from JPL Horizons Database.
Then ring_data contains multiple potentials of the rings of saturn, the two main ones are: ring_data_2025-01-08 that is a detailed and accurate potential of the current rings
and then there is ring_data_2025-01-09, a zero potenital used to turn of the ring potential in the code.
Lastly there is a simulation_data which contains all run simulations, these are not available here as this would take up to much drive space, we currently have arround 106.5 Gb of data.
For all but ring_data there are log files that include all saved files located within, including the import properties of these files.
Then the last main folder is SatrunMoonLibrary which includes most of all the code stack. It's mostly split up in 2 folders: SaturnMoonAnalysis and SaturnMoonSimulation.
SaturnMoonSimulation contains all the code files related to gathering and rotating the horizons_data, managing this data, and then running the simulation with this data.
the simulation code is written in C++ and is located within this folder in the SimulationFolder, with pybind11 a python module has been compiled to be able to run this simulation
within the python environment, this module is called simulation.pyd and is located in the main folder. Then to create this module pybind11, CMAKE are required and a Powershell script is used
to streamline the build process. Then on The Analysis side of things we have SaturnMoonAnalysis that includes the file data_class.py which includes the classes that describe the structure
of our datasets used for the Analysis and then there are other files that include classes pertaining to specific analysis purposes, all of whom take a dataset of the form of the data class 
structure as input. These specific classes include a classes with all animation functions stored and classes about the wave structure in saturn rings. Then to use all the code in the SatrunMoonLibrary
there are 2 main notebooks: simulation_notebook.ipynb and data_analysis.ipynb each respectively containing the executions of the the fucntions from SatrunMoonLibrary.
finally there is the notebook ringpotenial.ipynb that contains all the code used to create the ring potentials.

Dependencies:

To create the simulation.pyd file:
- CMAKE
- pybind11
- C++

To run the python files, we need the following libraries
- Astropy
- Astroquery
- Numpy
- Matplotlib
- Scikitlearn
- Scipy
- Pandas

Then we aslo depend on the JPL Horizons Database
link: https://ssd.jpl.nasa.gov/horizons/


NB: If there are any questions about the implementation feel free to ask.

