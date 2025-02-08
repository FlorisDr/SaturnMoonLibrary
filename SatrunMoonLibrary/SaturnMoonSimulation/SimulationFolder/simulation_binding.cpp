#define _USE_MATH_DEFINES
#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <array>
#include "simulation10.h"

namespace py = pybind11;

double run_simulation(
    double dt,
    int timesteps,
    int moon_count,
    int num_test_particles,
    int saved_points_modularity,
    int skipped_timesteps,
    double inner_radius,
    double outer_radius,
    double J2,
    std::vector<Body>& bodies,
    std::string& output_filename,
    std::string& R_matrix_filename,
    std::string& z_matrix_filename,
    int matrix_rows,
    int matrix_cols,
    int num_radial_bins,
    int num_azimuthal_bins,
    double theta_max,
    bool include_particle_moon_collisions,
    bool include_shear_forces,
    const std::string& integrator = "Leapfrog",
    const std::string& initialisation_method = "standard"
   
) {
    try {
        py::print("1: Initializing output file");
        std::ofstream file(output_filename, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Failed to open output file: " + output_filename);
        }

        py::print("2: Initializing simulation environment");
        std::vector<double> R(matrix_rows);
        std::vector<double> local_z(matrix_cols);
        for (int i = 0; i < matrix_rows; ++i) R[i] = 1e6 + i * (5e8 - 1e6) / (matrix_rows - 1);
        for (int i = 0; i < matrix_cols; ++i) local_z[i] = (i - 200) * 200e6 / (matrix_cols - 1);

        py::print("3: Loading matrix data");
        auto matrix_r = load_matrix_file(R_matrix_filename.c_str(), matrix_rows, matrix_cols);
        auto matrix_z = load_matrix_file(z_matrix_filename.c_str(), matrix_rows, matrix_cols);

        py::print("4: Initializing test particles");
        if (theta_max == 2 * M_PI && initialisation_method == "standard") {
            initialize_test_particles(bodies, num_test_particles, inner_radius, outer_radius);
        }
        else if (theta_max == 2 * M_PI && initialisation_method == "linear") {
            initialize_test_particles_linearly_spaced(bodies, num_test_particles, inner_radius, outer_radius);
        }
        else if (initialisation_method == "pizza slice" || integrator == "Leapfrog Pizza Slice") {
            initialize_test_particles_pizza_slice(bodies, num_test_particles, inner_radius, outer_radius, theta_max);
        }
        else {
            throw std::runtime_error("Unknown initialisation method: " + initialisation_method);
        }

        py::print("5: Starting simulation timing");
        auto start_time = std::chrono::high_resolution_clock::now();

        // Compute initial accelerations before starting integrastion method
        py::print("6: computing initial acceleration");
        compute_accelerations(bodies, matrix_r, matrix_z, matrix_rows, matrix_cols, R, local_z, J2, moon_count);


        py::print("7: Running simulation");

        for (int step = 0; step < timesteps; ++step) {
            if (integrator == "Leapfrog") {
                update_bodies_leapfrog(bodies, matrix_r, matrix_z, matrix_rows, matrix_cols, R, local_z, J2, moon_count, dt);
            }
            else if (integrator == "Leapfrog Pizza Slice") {
                update_bodies_leapfrog_pizza_slice(bodies, matrix_r, matrix_z, matrix_rows, matrix_cols, R, local_z, J2, moon_count, num_test_particles, dt, theta_max);
            }
            else if (integrator == "Yoshida") {
                update_bodies_yoshida(bodies, matrix_r, matrix_z, matrix_rows, matrix_cols, R, local_z, J2, moon_count, dt);
            }
            else if (integrator == "Yoshida Optimized") {
                update_bodies_yoshida_optimized(bodies, matrix_r, matrix_z, matrix_rows, matrix_cols, R, local_z, J2, moon_count, dt);
            }
            else if (integrator == "Yoshida 6th Order") {
                update_bodies_yoshida_6th_order(bodies, matrix_r, matrix_z, matrix_rows, matrix_cols, R, local_z, J2, moon_count, dt);
            }
            else if (integrator == "Yoshida 8th Order") {
                update_bodies_yoshida_8th_order(bodies, matrix_r, matrix_z, matrix_rows, matrix_cols, R, local_z, J2, moon_count, dt);
            }
            else if (integrator == "Euler") {
                update_bodies_euler(bodies, matrix_r, matrix_z, matrix_rows, matrix_cols, R, local_z, J2, moon_count, dt);
            }
            else if (integrator == "RK4") {
                update_bodies_rk4(bodies, matrix_r, matrix_z, matrix_rows, matrix_cols, R, local_z, J2, moon_count, dt);
            }
            else {
                throw std::runtime_error("Unknown integrator: " + integrator);
            }

            if (include_particle_moon_collisions) {
                handle_collisions(bodies, moon_count);
            }

            if (include_shear_forces) {
                py::print("deprecated function");
                //std::vector<double> radial_bins = initialize_radial_bins(inner_radius, outer_radius, num_radial_bins);
                //apply_shear_forces_hill_2d(bodies, radial_bins, num_azimuthal_bins, dt);
            }

            if (step % saved_points_modularity == 0 && step < skipped_timesteps) {
                py::print("Step: ", step, "/", timesteps, " not saved");
            }
            else if (step % saved_points_modularity == 0 && step >= skipped_timesteps) {
                py::print("Step: ", step, "/", timesteps, " saved");
                save_to_binary(output_filename, bodies);
            }
        }

        py::print("8: Final flush to ensure all data is saved");
        save_to_binary(output_filename, bodies, true);

        py::print("9: Ending timing and printing results");
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        py::print("Simulation completed in ", duration / 1000.0, " seconds");

        // Return the duration in seconds
        return duration / 1000.0;
    }
    catch (const std::exception& e) {
        py::print("Error: ", e.what());
        throw;
    }
}
PYBIND11_MODULE(simulation, m) {
    m.doc() = R"(
    N-body simulation module for modeling the interactions of celestial bodies under the influence of gravity.

    This module provides functionality for running a simulation of a system of bodies (planets, moons, and test particles) 
    using various numerical integration methods (e.g., Leapfrog, Yoshida, Euler, RK4). The simulation can account for gravitational 
    effects, collisions, shear forces, and other physics parameters such as the J2 perturbation of the primary body.

    Key features:
    - Supports multiple integrators including Leapfrog, Yoshida (4th, 6th, and 8th order), Euler, and RK4.
    - Customizable initial conditions and particle arrangements.
    - Support for particle-moon collisions.
    - Ability to save simulation results to binary files for further analysis.
    )";

    m.def("run_simulation", &run_simulation, R"(
    Run an N-body simulation of a system of celestial bodies over a specified number of timesteps.

    This function simulates the gravitational interactions between celestial bodies (such as planets, moons, and test particles),
    including effects like the J2 perturbation. Users can select different types of numerical integrators and initialization methods
    for the bodies involved in the simulation.

    Parameters:
        dt (float): The time step for the simulation (in seconds).
        timesteps (int): The total number of timesteps to simulate.
        moon_count (int): The number of moons in the simulation.
        num_test_particles (int): The number of test particles in the simulation.
        saved_points_modularity (int): The interval at which data points are saved (every 'saved_points_modularity' steps).
        skipped_timesteps (int): The number of timesteps to skip before saving any data.
        inner_radius (float): The inner radius for initializing test particles (in meters).
        outer_radius (float): The outer radius for initializing test particles (in meters).
        J2 (float): The J2 gravitational perturbation constant (default value for Saturn: -16290e-6).
        bodies (list of Body): A list of celestial bodies (e.g., planets, moons, test particles) in the simulation.
        output_filename (str): The name of the file where simulation results will be saved (in binary format).
        R_matrix_filename (str): The file containing the radial distances (R) matrix for the ring potential.
        z_matrix_filename (str): The file containing the vertical positions (z) matrix for the ring potential.
        matrix_rows (int): The number of rows in the R and z matrices.
        matrix_cols (int): The number of columns in the R and z matrices.
        num_radial_bins (int): The number of radial bins for collisions or shear.
        num_azimuthal_bins (int): The number of azimuthal bins for collisions or shear.
        theta_max (float): The maximum angular extent for initializing test particles (in radians).
        include_shear_forces (bool): Flag to include shear forces in the simulation (default is False).
        include_particle_moon_collisions (bool): Flag to include particle-moon collisions in the simulation (default is True).
        integrator (str): The numerical integration method to use. Available options:
            - "Leapfrog": Standard leapfrog integrator, suitable for long-term stability.
            - "Leapfrog Pizza Slice": Leapfrog with periodic boundaries set by theta_max.
            - "Euler": Simple Euler method, less accurate and not symplectic.
            - "RK4": Fourth-order Runge-Kutta method, more accurate but slower and not symplectic.
            - "Yoshida": 4th-order symplectic integrator for accurate simulations over long timescales.
            - "Yoshida Optimized": Optimized version of the 4th-order Yoshida integrator.
            - "Yoshida 6th Order": 6th-order symplectic Yoshida integrator.
            - "Yoshida 8th Order": 8th-order symplectic Yoshida integrator.
        initialisation_method (str): The method used to initialize test particles. Available options:
            - "standard": Random distribution of particles within the defined radii.
            - "linear": Linearly spaced initialization of particles.
            - "pizza slice": Initialization in a wedge shape (only valid when using the Pizza Slice integrator).
        return_filepath (bool): Whether to return the path to the generated output file (default is False).

    Returns:
        float: The total runtime of the simulation in seconds.
        str: The file path to the output file (if `return_filepath` is True).

    Example:
        >>> bodies = [Body("Earth", 5.97e24, 6371e3, [0, 0, 0], [0, 0, 0], True), ...]  # List of Body objects
        >>> simulation_time = run_simulation(
                1.0, 
                100000, 
                2, 
                100, 
                10, 
                100, 
                1e7, 
                4e7, 
                -16290e-6, 
                bodies, 
                'output.dat', 
                'R_matrix.txt', 
                'z_matrix.txt', 
                100, 
                100, 
                200, 
                360, 
                2 * M_PI, 
                False, 
                True, 
                'Leapfrog', 
                'standard'
            )
        Simulation completed in 300.5 seconds
   )",
        py::arg("dt"),
        py::arg("timesteps"),
        py::arg("moon_count"),
        py::arg("num_test_particles"),
        py::arg("saved_points_modularity"),
        py::arg("skipped_timesteps"),
        py::arg("inner_radius"),
        py::arg("outer_radius"),
        py::arg("J2") = -16290e-6,
        py::arg("bodies"),
        py::arg("output_filename"),
        py::arg("R_matrix_filename"),
        py::arg("z_matrix_filename"),
        py::arg("matrix_rows"),
        py::arg("matrix_cols"),
        py::arg("num_radial_bins") = 200,
        py::arg("num_azimuthal_bins") = 360,
        py::arg("theta_max"),
        py::arg("include_shear_forces") = false,
        py::arg("include_particle_moon_collisions") = true,
        py::arg("integrator") = "Leapfrog",
        py::arg("initialisation_method")
    );


    py::class_<Body>(m, "Body")
        .def(py::init<const std::string&, double, double, const std::array<double, 3> &, const std::array<double, 3> &, bool>())
        .def_readwrite("name", &Body::name)
        .def_readwrite("mass", &Body::mass)
        .def_readwrite("radius", &Body::radius)
        .def_readwrite("pos", &Body::pos)
        .def_readwrite("vel", &Body::vel)
        .def_readwrite("acc", &Body::acc)
        .def_readwrite("is_test_particle", &Body::is_test_particle);
}
