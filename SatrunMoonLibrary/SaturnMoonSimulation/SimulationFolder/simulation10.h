#pragma once
#ifndef SIMULATION10_H
#define SIMULATION10_H

#include <cmath>
#include <iostream>
#include <fstream>
#include <array>
#include <cmath>
#include <vector>
#include <utility>
#include <string>
#include <chrono>
#include <omp.h>
#include <unordered_map>
#include <algorithm>
#include <cstdint>
#include <stdexcept>

// Constants
const double G = 6.67430e-11; // Gravitational constant (m^3 kg^-1 s^-2)
const double dt = 600;    // Time step (seconds)
const int timesteps = 12000; // Number of time steps to simulate
//const double J2 = -16290e-6;  // J2 coefficient (dimensionless)
//const double Requa = 6.3781e6; // Equatorial radius of the central body in meters (e.g., Earth)
//const double J2_fconstant = J2 * Requa * Requa;
const int num_test_particles = 10000;
const int saved_points_modularity = 1;
const int skipped_timesteps = 8000;
const int moon_count = 20;
const double inner_radius = 115000 * 1000.0; // Inner boundary of the ring (meters)
const double outer_radius = 140000 * 1000.0; // Outer boundary of the ring (meters)

//const unsigned int MATRIX_ROWS = 500;
//const unsigned int MATRIX_COLS = 401;

// Declare `R` and `z` as extern so that they can be defined in a .cpp file
extern std::vector<double> R;  // Generated at main
extern std::vector<double> z;  // Generated at main

// Structure to represent a celestial body
struct Body {
    std::string name;
    double mass;                     // Mass (kg)
    double radius;                   // Radius (m)
    std::array<double, 3> pos;       // Position {x, y, z} (meters)
    std::array<double, 3> vel;       // Velocity {vx, vy, vz} (m/s)
    std::array<double, 3> acc = { 0 }; // Acceleration {ax, ay, az} (m/s^2)
    bool is_test_particle = false;   // Flag to indicate if it's a test particle

    Body(const std::string& n, double m, double r, const std::array<double, 3>& p, const std::array<double, 3>& v, bool test_particle = false)
        : name(n), mass(m), radius(r), pos(p), vel(v), is_test_particle(test_particle) {
    }
};

// Function Prototypes
std::vector<std::vector<double>> load_matrix_file(const char* filename, unsigned int rows, unsigned int cols);
std::array<double, 3> ring_acceleration_cartesian(
    std::array<double, 3> position,
    const std::vector<std::vector<double>>& matrix_r,
    const std::vector<std::vector<double>>& matrix_z,
    const unsigned int matrix_rows,
    const unsigned int matrix_cols,
    const std::vector<double>& R,
    const std::vector<double>& z);
void compute_accelerations(
    std::vector<Body>& bodies,
    const std::vector<std::vector<double>>& matrix_r,
    const std::vector<std::vector<double>>& matrix_z,
    const unsigned int matrix_rows,
    const unsigned int matrix_cols,
    const std::vector<double>& R,
    const std::vector<double>& z,
    const double J2,
    const int moon_count);
void handle_collisions(std::vector<Body>& bodies, const int moon_count);
void update_bodies_leapfrog(
    std::vector<Body>& bodies,
    const std::vector<std::vector<double>>& matrix_r,
    const std::vector<std::vector<double>>& matrix_z,
    const unsigned int matrix_rows,
    const unsigned int matrix_cols,
    const std::vector<double>& R,
    const std::vector<double>& z,
    const double J2,
    const int moon_count, const double dt
);


void update_bodies_leapfrog_pizza_slice(
    std::vector<Body>& bodies,
    const std::vector<std::vector<double>>& matrix_r,
    const std::vector<std::vector<double>>& matrix_z,
    const unsigned int matrix_rows,
    const unsigned int matrix_cols,
    const std::vector<double>& R,
    const std::vector<double>& z,
    const double J2,
    const int moon_count, const int number_of_test_particles,
    const double dt,
    const double theta_max,               // Add parameter for angular boundary
    const bool moons_rotate_freely = true        // New parameter to allow moons to rotate freely
);
void update_bodies_yoshida(
    std::vector<Body>& bodies,
    const std::vector<std::vector<double>>& matrix_r,
    const std::vector<std::vector<double>>& matrix_z,
    const unsigned int matrix_rows,
    const unsigned int matrix_cols,
    const std::vector<double>& R,
    const std::vector<double>& z,
    const double J2,
    const int moon_count,
    const double dt);
void update_bodies_yoshida_optimized(
    std::vector<Body>& bodies,
    const std::vector<std::vector<double>>& matrix_r,
    const std::vector<std::vector<double>>& matrix_z,
    const unsigned int matrix_rows,
    const unsigned int matrix_cols,
    const std::vector<double>& R,
    const std::vector<double>& z,
    const double J2,
    const int moon_count,
    const double dt);
void update_bodies_yoshida_6th_order(
    std::vector<Body>& bodies,
    const std::vector<std::vector<double>>& matrix_r,
    const std::vector<std::vector<double>>& matrix_z,
    const unsigned int matrix_rows,
    const unsigned int matrix_cols,
    const std::vector<double>& R,
    const std::vector<double>& z,
    const double J2,
    const int moon_count,
    const double dt);

void update_bodies_yoshida_8th_order(
    std::vector<Body>& bodies,
    const std::vector<std::vector<double>>& matrix_r,
    const std::vector<std::vector<double>>& matrix_z,
    const unsigned int matrix_rows,
    const unsigned int matrix_cols,
    const std::vector<double>& R,
    const std::vector<double>& z,
    const double J2,
    const int moon_count,
    const double dt);

void update_bodies_euler(
    std::vector<Body>& bodies,
    const std::vector<std::vector<double>>& matrix_r,
    const std::vector<std::vector<double>>& matrix_z,
    const unsigned int matrix_rows,
    const unsigned int matrix_cols,
    const std::vector<double>& R,
    const std::vector<double>& z,
    const double J2,
    const int moon_count,
    const double dt);

void update_bodies_rk4(
    std::vector<Body>& bodies,
    const std::vector<std::vector<double>>& matrix_r,
    const std::vector<std::vector<double>>& matrix_z,
    const unsigned int matrix_rows,
    const unsigned int matrix_cols,
    const std::vector<double>& R,
    const std::vector<double>& z,
    const double J2,
    const int moon_count,
    const double dt);

void save_to_binary(const std::string& filename, const std::vector<Body>& bodies, bool flush_all = false);
void initialize_test_particles(std::vector<Body>& bodies, const int num_particles, const double inner_radius, const double outer_radius);

void initialize_test_particles_linearly_spaced(
    std::vector<Body>& bodies,
    const int num_particles,
    const double inner_radius,
    const double outer_radius
);

void initialize_test_particles_pizza_slice(
    std::vector<Body>& bodies,
    const int num_particles,
    const double inner_radius,
    const double outer_radius,
    const double theta_max // New boundary parameter for angular segment
);

#endif // SIMULATION10_H
