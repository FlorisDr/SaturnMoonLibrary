#define _USE_MATH_DEFINES
#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <utility>
#include <array>
#include <vector>
#include <string>
#include <chrono>
#include <omp.h> // For parallelization^
#include <unordered_map>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include "simulation10.h"

namespace py = pybind11;

// void check_particle_collision(Body& p1, Body& p2) {
//     double dx = p1.pos[0] - p2.pos[0];
//     double dy = p1.pos[1] - p2.pos[1];
//     double dz = p1.pos[2] - p2.pos[2];
//     const double max_interaction_distance = 1e4; // Define a threshold for interactions
//     if (std::abs(dx) > max_interaction_distance || std::abs(dy) > max_interaction_distance || std::abs(dz) > max_interaction_distance) {
//         return; // Skip if the particles are too far apart
//     }
//     double distance_sq = dx * dx + dy * dy + dz * dz;
//     const double part_dist = 4e6; // Assume radius is 1 km
//     if (distance_sq < part_dist) {
//         // Elastic collision
//         std::array<double, 3> diff = {dx, dy, dz};
//         std::array<double, 3> relative_vel = {p1.vel[0] - p2.vel[0], p1.vel[1] - p2.vel[1], p1.vel[2] - p2.vel[2]};
//         double distance = std::sqrt(distance_sq);
//         double dot_product = 0.0;
//         for (int k = 0; k < 3; ++k) {
//             dot_product += relative_vel[k] * diff[k];
//         }
//         double factor = (2 * p2.mass / (p1.mass + p2.mass)) * dot_product / distance_sq;
//         for (int k = 0; k < 3; ++k) {
//             p1.vel[k] -= factor * diff[k];
//             p2.vel[k] += factor * diff[k];
//         }
//     }
// };

// void handle_collisions_particles(std::vector<Body>& bodies) {
//     for (size_t i=19; i<bodies.size(); i++) {
//         for (size_t j=19; j<bodies.size(); j++) {
//             check_particle_collision(bodies[i],bodies[j]);
//         }      
//     }
// }

// Loads .bin file as matrix for ring potential
std::vector<std::vector<double>> load_matrix_file(const char* filename, unsigned int rows, unsigned int cols) {
    std::ifstream file(filename, std::ios::binary);
    if (!file)
        throw std::invalid_argument("Could not open file");

    // Read the matrix
    std::vector<std::vector<double>> matrix(rows, std::vector<double>(cols));
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            double value;
            file.read(reinterpret_cast<char*>(&value), sizeof(double));

            if (file.eof()) {
                throw std::invalid_argument("Unexpected end of file");
            }

            matrix[i][j] = value;
        }
    }

    file.close();
    return matrix;
}

// Calculates acceleration due to Saturn's rings, only applicable if 0 < r < 5 * 10^8 and -100 * 10^6 < z < 100 * 10^6
std::array<double, 3> ring_acceleration_cartesian(
    std::array<double, 3> position,
    const std::vector<std::vector<double>>& matrix_r,
    const std::vector<std::vector<double>>& matrix_z,
    const unsigned int matrix_rows,
    const unsigned int matrix_cols,
    const std::vector<double>& R,
    const std::vector<double>& z)
{
    double cur_R = std::sqrt(position[0] * position[0] + position[1] * position[1]);

    // Return empty vector since we are outside the matrix
    if (cur_R < 1e6 || cur_R > 5e8 || position[2] < -100e6 || position[2] > 100e6) {
        return { 0, 0, 0 };
    }

    double calc_dR = (5e8 - 1e6) / static_cast<double>(matrix_rows - 1);
    double calc_dz = 200e6 / static_cast<double>(matrix_cols - 1);

    int index_R_first = static_cast<int>(std::floor((cur_R - 1e6) / calc_dR));
    int index_R_second = static_cast<int>(std::ceil((cur_R - 1e6) / calc_dR));
    int index_z_first = static_cast<int>(std::floor(position[2] / calc_dz)) + (matrix_rows - 1) / 2;
    int index_z_second = static_cast<int>(std::ceil(position[2] / calc_dz)) + (matrix_rows - 1) / 2;

    // linearly interpolate between points, use 2d linearization
    double interp_R = (cur_R - R[index_R_first]) / calc_dR;
    double interp_z = (position[2] - z[index_z_first]) / calc_dz;

    double cur_forces_R = (matrix_r[index_R_first][index_z_first] + matrix_r[index_R_second][index_z_first]
        + matrix_r[index_R_first][index_z_second] + matrix_r[index_R_second][index_z_second]) / 4.0;
    double cur_forces_z = (matrix_z[index_R_first][index_z_first] + matrix_z[index_R_second][index_z_first]
        + matrix_z[index_R_first][index_z_second] + matrix_z[index_R_second][index_z_second]) / 4.0;

    return {
        position[0] / cur_R * cur_forces_R,
        position[1] / cur_R * cur_forces_R,
        cur_forces_z
    };
}

// Function to compute accelerations due to gravity with J2 effect
void compute_accelerations(
    std::vector<Body>& bodies,
    const std::vector<std::vector<double>>& matrix_r,
    const std::vector<std::vector<double>>& matrix_z,
    const unsigned int matrix_rows,
    const unsigned int matrix_cols,
    const std::vector<double>& R,
    const std::vector<double>& z,
    const double J2,
    const int moon_count)
{
    // J2 related parameters
    constexpr double Requa = 6.3781e6; // Equatorial radius of the central body in meters (e.g., Earth)
    double J2_fconstant = J2 * Requa * Requa;

    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(bodies.size()); ++i) {
        if (bodies[i].mass<1) continue;
        
        // Reset acceleration for body i
        // in essence we do acc={0,0,0}; acc += ring_acc so we can just do acc = ring_acc
        // bodies[i].acc = { 0, 0, 0 };

        std::array<double, 3> center = { bodies[i].pos[0] - bodies[0].pos[0],
                                          bodies[i].pos[1] - bodies[0].pos[1],
                                          bodies[i].pos[2] - bodies[0].pos[2] };

        bodies[i].acc = ring_acceleration_cartesian(center, matrix_r, matrix_z, matrix_rows, matrix_cols, R, z);

        //if (bodies[i].is_test_particle) continue; // Test particles don't exert gravitational forces

        for (size_t j = 0; j < moon_count; ++j) {
            if (i == j) continue;

            std::array<double, 3> diff = {bodies[j].pos[0] - bodies[i].pos[0],
                                          bodies[j].pos[1] - bodies[i].pos[1],
                                          bodies[j].pos[2] - bodies[i].pos[2]};

            double distance_sq = diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2];
            double distance = std::sqrt(distance_sq);
            if (distance == 0.0) continue;

            // is acceleration due to gravity, not force
            double force = G * bodies[j].mass / (distance_sq * distance);

            // J2 correction
            double r2 = distance_sq;
            double r = distance;
            double j2_factor = 1.0;
            double j2_factor_z = 1.0;
            if (j==0) { // Apply J2 only for main bodies
                double k = J2_fconstant / r2;
                j2_factor += k * (7.5F * (diff[2] * diff[2] / r2) - 1.5F);
                j2_factor_z += j2_factor - k * 3.0F;
                //j2_factor += (Requa * Requa / r2) * J2 * (15.0 / 2.0 * (diff[2] * diff[2] / r2) - 3.0 / 2.0); // 1.5F
                //j2_factor_z += (Requa * Requa / r2) * J2 * (15.0 / 2.0 * (diff[2] * diff[2] / r2) - 9.0 / 2.0); // 4.5F
            }

            bodies[i].acc[0] += force * diff[0] * j2_factor;
            bodies[i].acc[1] += force * diff[1] * j2_factor;
            bodies[i].acc[2] += force * diff[2] * j2_factor_z;
        }
    }
}


void handle_collisions(std::vector<Body>& bodies, const int moon_count) {
    const double out_of_sight_distance = 1e12; // Distance to move particles out of sight
    for (size_t i = moon_count; i < bodies.size(); ++i) {
        if (bodies[i].mass<1) continue;
        // Check for collisions with moons
        for (size_t j = 0; j < moon_count; ++j) {
            double dx = bodies[i].pos[0] - bodies[j].pos[0];
            double dy = bodies[i].pos[1] - bodies[j].pos[1];
            double dz = bodies[i].pos[2] - bodies[j].pos[2];
            double distance = std::sqrt(dx * dx + dy * dy + dz * dz);

            if (distance < bodies[j].radius) {
                // Move the particle out of sight
                bodies[i].mass = 0;
                bodies[i].pos = {out_of_sight_distance + bodies[i].pos[0], out_of_sight_distance + bodies[i].pos[1], out_of_sight_distance + bodies[i].pos[2]};
                bodies[i].vel = {0.0, 0.0, 0.0}; // Optionally stop its motion
                bodies[i].acc = { 0.0,0.0,0.0 };
                break;
            }
        }
    }
}
// Function to perform Leapfrog integration updates
void update_bodies_leapfrog(
    std::vector<Body>& bodies,
    const std::vector<std::vector<double>>& matrix_r,
    const std::vector<std::vector<double>>& matrix_z,
    const unsigned int matrix_rows,
    const unsigned int matrix_cols,
    const std::vector<double>& R,
    const std::vector<double>& z,
    const double J2,
    const int moon_count, const double dt)
{
    //py::print("first step of leapfrog");
    // First step: Update velocities by half a step
#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(bodies.size()); ++i) {
        for (int k = 0; k < 3; ++k) {
            bodies[i].vel[k] += 0.5 * bodies[i].acc[k] * dt;
        }
    }
    //py::print("done with first step of leapfrog");

    //py::print("second step of leapfrog");
    // Second step: Update positions using half-step velocities
#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(bodies.size()); ++i) {
        for (int k = 0; k < 3; ++k) {
            bodies[i].pos[k] += bodies[i].vel[k] * dt;
        }
    }
    //py::print("done with second step of leapfrog");

    // Recompute accelerations after position updates
    compute_accelerations(bodies, matrix_r, matrix_z, matrix_rows, matrix_cols, R, z, J2, moon_count);

    //py::print("final step of leapfrog");
    // Final step: Update velocities by the second half-step
#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(bodies.size()); ++i) {
        for (int k = 0; k < 3; ++k) {
            bodies[i].vel[k] += 0.5 * bodies[i].acc[k] * dt;
        }
    }
    //py::print("done with final step of leapfrog");
}


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
    const bool moons_rotate_freely        // New parameter to allow moons to rotate freely
) {
    // First step: Update velocities by half a step
#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(bodies.size()); ++i) {
        for (int k = 0; k < 3; ++k) {
            bodies[i].vel[k] += 0.5 * bodies[i].acc[k] * dt;
        }
    }

    // Second step: Update positions using half-step velocities
#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(bodies.size()); ++i) {
        // Update positions
        for (int k = 0; k < 3; ++k) {
            bodies[i].pos[k] += bodies[i].vel[k] * dt;
        }

        // Moons (bodies 0 to moon_count-1) rotate freely if moons_rotate_freely is true
        if (i < moon_count && moons_rotate_freely) {
            continue; // Moons are not subject to the periodic boundary condition
        }

        // For test particles, apply periodic boundary condition
        double angle = std::atan2(bodies[i].pos[1], bodies[i].pos[0]);
        double radius = std::sqrt(bodies[i].pos[0] * bodies[i].pos[0] + bodies[i].pos[1] * bodies[i].pos[1]);

        if (angle < 0) {
            angle += 2 * M_PI;  // Normalize angle to [0, 2π]
        }

        // If particle crosses the angular boundary (theta_max)
        if (angle > theta_max) {
            // Wrap angle based on the maximum theta
            double old_angle = angle;
            angle = std::fmod(angle, theta_max);

            // Calculate the change in angle
            double delta_angle = old_angle - angle;

            // Update the particle's position to reflect the periodic boundary
            bodies[i].pos[0] = radius * std::cos(angle);
            bodies[i].pos[1] = radius * std::sin(angle);

            // Now rotate the velocity and acceleration vectors by -delta_angle
            // 2D rotation matrix for velocity and acceleration (counterclockwise)
            double cos_delta = std::cos(-delta_angle);
            double sin_delta = std::sin(-delta_angle);

            // Rotate velocity
            double v_x = bodies[i].vel[0];
            double v_y = bodies[i].vel[1];
            bodies[i].vel[0] = v_x * cos_delta - v_y * sin_delta;
            bodies[i].vel[1] = v_x * sin_delta + v_y * cos_delta;

            // Rotate acceleration
            double a_x = bodies[i].acc[0];
            double a_y = bodies[i].acc[1];
            bodies[i].acc[0] = a_x * cos_delta - a_y * sin_delta;
            bodies[i].acc[1] = a_x * sin_delta + a_y * cos_delta;
        }
    }

    // Recompute accelerations after position updates
    compute_accelerations(bodies, matrix_r, matrix_z, matrix_rows, matrix_cols, R, z, J2, moon_count+number_of_test_particles);

    // Final step: Update velocities by the second half-step
#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(bodies.size()); ++i) {
        for (int k = 0; k < 3; ++k) {
            bodies[i].vel[k] += 0.5 * bodies[i].acc[k] * dt;
        }
    }
}


// Function to perform yoshida's 4th order integration updates
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
    const double dt)
{
    // Yoshida coefficients
    const double w0 = -std::cbrt(2) / (2 - std::cbrt(2));
    const double w1 = 1.0 / (2 - std::cbrt(2));
    const double c1 = w1 / 2.0;
    const double c4 = w1 / 2.0;
    const double c2 = (w0 + w1) / 2.0;
    const double c3 = c2;
    const double d1 = w1;
    const double d3 = w1;
    const double d2 = w0;

    // First intermediary step
#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(bodies.size()); ++i) {
        for (int k = 0; k < 3; ++k) {
            bodies[i].pos[k] += c1 * bodies[i].vel[k] * dt;  // First step
        }
    }


    compute_accelerations(bodies, matrix_r, matrix_z, matrix_rows, matrix_cols, R, z, J2, moon_count);

#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(bodies.size()); ++i) {
        for (int k = 0; k < 3; ++k) {
            bodies[i].vel[k] += d1 * bodies[i].acc[k] * dt;  // First velocity update
        }
    }

    // Second intermediary step
#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(bodies.size()); ++i) {
        for (int k = 0; k < 3; ++k) {
            bodies[i].pos[k] += c2 * bodies[i].vel[k] * dt;  // Second step
        }
    }

    compute_accelerations(bodies, matrix_r, matrix_z, matrix_rows, matrix_cols, R, z, J2, moon_count);

#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(bodies.size()); ++i) {
        for (int k = 0; k < 3; ++k) {
            bodies[i].vel[k] += d2 * bodies[i].acc[k] * dt;  // Second velocity update
        }
    }

    // Third intermediary step
#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(bodies.size()); ++i) {
        for (int k = 0; k < 3; ++k) {
            bodies[i].pos[k] += c3 * bodies[i].vel[k] * dt;  // Update positions
        }
    }

    compute_accelerations(bodies, matrix_r, matrix_z, matrix_rows, matrix_cols, R, z, J2, moon_count);

#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(bodies.size()); ++i) {
        for (int k = 0; k < 3; ++k) {
            bodies[i].vel[k] += d3 * bodies[i].acc[k] * dt;  // Update velocities
        }
    }

    // Final step: update positions using the last position coefficient (c4)
#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(bodies.size()); ++i) {
        for (int k = 0; k < 3; ++k) {
            bodies[i].pos[k] += c4 * bodies[i].vel[k] * dt;  // Final position update
        }
    }
}

// Function to perform yoshida's 4th order integration updates, optimized
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
    const double dt)
{
    // Yoshida constants
    constexpr double w0 = -1.091123635971721;
    constexpr double w1 = 1.784492467190112;
    constexpr double c1 = w1 / 2.0, c4 = w1 / 2.0;
    constexpr double c2 = (w0 + w1) / 2.0, c3 = c2;
    constexpr double d1 = w1, d3 = w1, d2 = w0;

    // Helper lambda for updating positions and velocities
    auto yoshida_update_positions = [&](double c) {
#pragma omp parallel for
        for (int i = 0; i < static_cast<int>(bodies.size()); ++i) {
            for (int k = 0; k < 3; ++k) {
                bodies[i].pos[k] += c * bodies[i].vel[k] * dt;
            }
        }
        };

    auto yoshida_update_velocities = [&](double d) {
#pragma omp parallel for
        for (int i = 0; i < static_cast<int>(bodies.size()); ++i) {
            for (int k = 0; k < 3; ++k) {
                bodies[i].vel[k] += d * bodies[i].acc[k] * dt;
            }
        }
        };

    // First intermediary step
    yoshida_update_positions(c1);
    compute_accelerations(bodies, matrix_r, matrix_z, matrix_rows, matrix_cols, R, z, J2, moon_count);
    yoshida_update_velocities(d1);

    // Second intermediary step
    yoshida_update_positions(c2);
    compute_accelerations(bodies, matrix_r, matrix_z, matrix_rows, matrix_cols, R, z, J2, moon_count);
    yoshida_update_velocities(d2);

    // Third intermediary step
    yoshida_update_positions(c3);
    compute_accelerations(bodies, matrix_r, matrix_z, matrix_rows, matrix_cols, R, z, J2, moon_count);
    yoshida_update_velocities(d3);

    // Final step: update positions using the last coefficient (c4)
    yoshida_update_positions(c4);
}

// Function to perform 6th-order Yoshida integration updates
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
    const double dt)
{
    constexpr double w6_1 = -1.17767998417887100694641568;
    constexpr double w6_2 = 0.23557321335935813368479318;
    constexpr double w6_3 = 0.78451361047755726381949763;
    constexpr double w6_0 = 1 - 2 * (w6_1 + w6_2 + w6_3);
    constexpr double c6_1 = w6_3 / 2.0, c6_8 = w6_3 / 2.0;
    constexpr double c6_2 = (w6_3 + w6_2) / 2.0, c6_7 = c6_2;
    constexpr double c6_3 = (w6_2 + w6_1) / 2.0, c6_6 = c6_3;
    constexpr double c6_4 = (w6_1 + w6_0) / 2.0, c6_5 = c6_4;
    constexpr double d6_1 = w6_3, d6_7 = w6_3;
    constexpr double d6_2 = w6_2, d6_6 = w6_2;
    constexpr double d6_3 = w6_1, d6_5 = w6_1;
    constexpr double d6_4 = w6_0;

        auto yoshida_update_positions = [&](double c) {
#pragma omp parallel for
        for (int i = 0; i < static_cast<int>(bodies.size()); ++i) {
            for (int k = 0; k < 3; ++k) {
                bodies[i].pos[k] += c * bodies[i].vel[k] * dt;
            }
        }
        };

    auto yoshida_update_velocities = [&](double d) {
#pragma omp parallel for
        for (int i = 0; i < static_cast<int>(bodies.size()); ++i) {
            for (int k = 0; k < 3; ++k) {
                bodies[i].vel[k] += d * bodies[i].acc[k] * dt;
            }
        }
        };

    // 7-stage update
    yoshida_update_positions(c6_1);
    compute_accelerations(bodies, matrix_r, matrix_z, matrix_rows, matrix_cols, R, z, J2, moon_count);
    yoshida_update_velocities(d6_1);

    yoshida_update_positions(c6_2);
    compute_accelerations(bodies, matrix_r, matrix_z, matrix_rows, matrix_cols, R, z, J2, moon_count);
    yoshida_update_velocities(d6_2);

    yoshida_update_positions(c6_3);
    compute_accelerations(bodies, matrix_r, matrix_z, matrix_rows, matrix_cols, R, z, J2, moon_count);
    yoshida_update_velocities(d6_3);

    yoshida_update_positions(c6_4);
    compute_accelerations(bodies, matrix_r, matrix_z, matrix_rows, matrix_cols, R, z, J2, moon_count);
    yoshida_update_velocities(d6_4);

    yoshida_update_positions(c6_5);
    compute_accelerations(bodies, matrix_r, matrix_z, matrix_rows, matrix_cols, R, z, J2, moon_count);
    yoshida_update_velocities(d6_5);

    yoshida_update_positions(c6_6);
    compute_accelerations(bodies, matrix_r, matrix_z, matrix_rows, matrix_cols, R, z, J2, moon_count);
    yoshida_update_velocities(d6_6);

    yoshida_update_positions(c6_7);
    compute_accelerations(bodies, matrix_r, matrix_z, matrix_rows, matrix_cols, R, z, J2, moon_count);
    yoshida_update_velocities(d6_7);

    yoshida_update_positions(c6_8);
}

// Function to perform 8th-order Yoshida integration updates
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
    const double dt)
{
    constexpr double w8_1 = -0.161582374150097E1;
    constexpr double w8_2 = -0.244699182370524E1;
    constexpr double w8_3 = -0.716989419708120E-2;
    constexpr double w8_4 = 0.244002732616735E1;
    constexpr double w8_5 = 0.1577399281236E0;
    constexpr double w8_6 = 0.18202063097071E1;
    constexpr double w8_7 = 0.10424262086999E1;
    constexpr double w8_0 = 1 - 2 * (w8_1 + w8_2 + w8_3 + w8_4 + w8_5 + w8_6 + w8_7);
    constexpr double c8_1 = w8_7 / 2.0, c8_16 = c8_1;
    constexpr double c8_2 = (w8_7 + w8_6) / 2.0, c8_15 = c8_2;
    constexpr double c8_3 = (w8_6 + w8_5) / 2.0, c8_14 = c8_3;
    constexpr double c8_4 = (w8_5 + w8_4) / 2.0, c8_13 = c8_4;
    constexpr double c8_5 = (w8_4 + w8_3) / 2.0, c8_12 = c8_5;
    constexpr double c8_6 = (w8_3 + w8_2) / 2.0, c8_11 = c8_6;
    constexpr double c8_7 = (w8_2 + w8_1) / 2.0, c8_10 = c8_7;
    constexpr double c8_8 = (w8_1 + w8_0) / 2.0, c8_9 = c8_8;
    constexpr double d8_1 = w8_7, d8_15 = w8_7;
    constexpr double d8_2 = w8_6, d8_14 = w8_6;
    constexpr double d8_3 = w8_5, d8_13 = w8_5;
    constexpr double d8_4 = w8_4, d8_12 = w8_4;
    constexpr double d8_5 = w8_3, d8_11 = w8_3;
    constexpr double d8_6 = w8_2, d8_10 = w8_2;
    constexpr double d8_7 = w8_1, d8_9 = w8_1;
    constexpr double d8_8 = w8_0;

    auto yoshida_update_positions = [&](double c) {
#pragma omp parallel for
        for (int i = 0; i < static_cast<int>(bodies.size()); ++i) {
            for (int k = 0; k < 3; ++k) {
                bodies[i].pos[k] += c * bodies[i].vel[k] * dt;
            }
        }
        };

    auto yoshida_update_velocities = [&](double d) {
#pragma omp parallel for
        for (int i = 0; i < static_cast<int>(bodies.size()); ++i) {
            for (int k = 0; k < 3; ++k) {
                bodies[i].vel[k] += d * bodies[i].acc[k] * dt;
            }
        }
        };

    // 15-stage update
    yoshida_update_positions(c8_1);
    compute_accelerations(bodies, matrix_r, matrix_z, matrix_rows, matrix_cols, R, z, J2, moon_count);
    yoshida_update_velocities(d8_1);

    yoshida_update_positions(c8_2);
    compute_accelerations(bodies, matrix_r, matrix_z, matrix_rows, matrix_cols, R, z, J2, moon_count);
    yoshida_update_velocities(d8_2);

    yoshida_update_positions(c8_3);
    compute_accelerations(bodies, matrix_r, matrix_z, matrix_rows, matrix_cols, R, z, J2, moon_count);
    yoshida_update_velocities(d8_3);

    yoshida_update_positions(c8_4);
    compute_accelerations(bodies, matrix_r, matrix_z, matrix_rows, matrix_cols, R, z, J2, moon_count);
    yoshida_update_velocities(d8_4);

    yoshida_update_positions(c8_5);
    compute_accelerations(bodies, matrix_r, matrix_z, matrix_rows, matrix_cols, R, z, J2, moon_count);
    yoshida_update_velocities(d8_5);

    yoshida_update_positions(c8_6);
    compute_accelerations(bodies, matrix_r, matrix_z, matrix_rows, matrix_cols, R, z, J2, moon_count);
    yoshida_update_velocities(d8_6);

    yoshida_update_positions(c8_7);
    compute_accelerations(bodies, matrix_r, matrix_z, matrix_rows, matrix_cols, R, z, J2, moon_count);
    yoshida_update_velocities(d8_7);

    yoshida_update_positions(c8_8);
    compute_accelerations(bodies, matrix_r, matrix_z, matrix_rows, matrix_cols, R, z, J2, moon_count);
    yoshida_update_velocities(d8_8);

    yoshida_update_positions(c8_9);
    compute_accelerations(bodies, matrix_r, matrix_z, matrix_rows, matrix_cols, R, z, J2, moon_count);
    yoshida_update_velocities(d8_9);

    yoshida_update_positions(c8_10);
    compute_accelerations(bodies, matrix_r, matrix_z, matrix_rows, matrix_cols, R, z, J2, moon_count);
    yoshida_update_velocities(d8_10);

    yoshida_update_positions(c8_11);
    compute_accelerations(bodies, matrix_r, matrix_z, matrix_rows, matrix_cols, R, z, J2, moon_count);
    yoshida_update_velocities(d8_11);

    yoshida_update_positions(c8_12);
    compute_accelerations(bodies, matrix_r, matrix_z, matrix_rows, matrix_cols, R, z, J2, moon_count);
    yoshida_update_velocities(d8_12);

    yoshida_update_positions(c8_13);
    compute_accelerations(bodies, matrix_r, matrix_z, matrix_rows, matrix_cols, R, z, J2, moon_count);
    yoshida_update_velocities(d8_13);

    yoshida_update_positions(c8_14);
    compute_accelerations(bodies, matrix_r, matrix_z, matrix_rows, matrix_cols, R, z, J2, moon_count);
    yoshida_update_velocities(d8_14);

    yoshida_update_positions(c8_15);
    compute_accelerations(bodies, matrix_r, matrix_z, matrix_rows, matrix_cols, R, z, J2, moon_count);
    yoshida_update_velocities(d8_15);

    yoshida_update_positions(c8_16);
}

// Function to perform Euler integration updates
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
    const double dt)
{
    // First step: Update positions based on current velocities
#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(bodies.size()); ++i) {
        for (int k = 0; k < 3; ++k) {
            bodies[i].pos[k] += bodies[i].vel[k] * dt;
        }
    }

    // Recompute accelerations after position updates
    compute_accelerations(bodies, matrix_r, matrix_z, matrix_rows, matrix_cols, R, z, J2, moon_count);

    // Second step: Update velocities based on current accelerations
#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(bodies.size()); ++i) {
        for (int k = 0; k < 3; ++k) {
            bodies[i].vel[k] += bodies[i].acc[k] * dt;
        }
    }
}

// Function to perform RK4 integration updates
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
    const double dt)
{
    // Vectors to store intermediate positions and velocities (k1, k2, k3, k4)
    std::vector<std::array<std::array<double, 3>, 4>> k_pos(bodies.size()), k_vel(bodies.size());

    // Compute k1 (initial step)
#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(bodies.size()); ++i) {
        for (int k = 0; k < 3; ++k) {
            k_pos[i][0][k] = bodies[i].vel[k];  // k1 for position
            k_vel[i][0][k] = bodies[i].acc[k];  // k1 for velocity (acceleration)
        }
    }

    // Helper lambda to update position and velocity for intermediate steps
    auto rk_update_intermediate = [&](std::vector<Body>& temp_bodies, int step, double factor) {
#pragma omp parallel for
        for (int i = 0; i < static_cast<int>(temp_bodies.size()); ++i) {
            for (int k = 0; k < 3; ++k) {
                temp_bodies[i].pos[k] = bodies[i].pos[k] + factor * k_pos[i][step - 1][k] * dt;
                temp_bodies[i].vel[k] = bodies[i].vel[k] + factor * k_vel[i][step - 1][k] * dt;
            }
        }
        };

    // Create a temporary bodies vector to hold intermediate positions and velocities
    std::vector<Body> temp_bodies = bodies;

    // Compute k2
    rk_update_intermediate(temp_bodies, 1, 0.5);
    compute_accelerations(temp_bodies, matrix_r, matrix_z, matrix_rows, matrix_cols, R, z, J2, moon_count);
#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(bodies.size()); ++i) {
        for (int k = 0; k < 3; ++k) {
            k_pos[i][1][k] = temp_bodies[i].vel[k];  // k2 for position
            k_vel[i][1][k] = temp_bodies[i].acc[k];  // k2 for velocity
        }
    }

    // Compute k3
    rk_update_intermediate(temp_bodies, 2, 0.5);
    compute_accelerations(temp_bodies, matrix_r, matrix_z, matrix_rows, matrix_cols, R, z, J2, moon_count);
#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(bodies.size()); ++i) {
        for (int k = 0; k < 3; ++k) {
            k_pos[i][2][k] = temp_bodies[i].vel[k];  // k3 for position
            k_vel[i][2][k] = temp_bodies[i].acc[k];  // k3 for velocity
        }
    }

    // Compute k4
    rk_update_intermediate(temp_bodies, 3, 1.0);
    compute_accelerations(temp_bodies, matrix_r, matrix_z, matrix_rows, matrix_cols, R, z, J2, moon_count);
#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(bodies.size()); ++i) {
        for (int k = 0; k < 3; ++k) {
            k_pos[i][3][k] = temp_bodies[i].vel[k];  // k4 for position
            k_vel[i][3][k] = temp_bodies[i].acc[k];  // k4 for velocity
        }
    }

    // Final update to positions and velocities using weighted average of k1, k2, k3, and k4
#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(bodies.size()); ++i) {
        for (int k = 0; k < 3; ++k) {
            bodies[i].pos[k] += (dt / 6.0) * (k_pos[i][0][k] + 2 * k_pos[i][1][k] + 2 * k_pos[i][2][k] + k_pos[i][3][k]);
            bodies[i].vel[k] += (dt / 6.0) * (k_vel[i][0][k] + 2 * k_vel[i][1][k] + 2 * k_vel[i][2][k] + k_vel[i][3][k]);
        }
    }
}






// Optimized function to save positions and velocities to a binary file
void save_to_binary(const std::string& filename, const std::vector<Body>& bodies, bool flush_all) {
    static std::vector<char> buffer;
    static const size_t buffer_limit = 1024 * 1024; // 1 MB buffer size

    // print statement
    std::cout << "Attempting to open file: " << filename << std::endl;
    // Open the file in append mode
    std::ofstream file;
    file.open(filename, std::ios::binary | std::ios::app);

    // Check if the file was opened successfully
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }
    for (const auto& body : bodies) {
        const char* pos_data = reinterpret_cast<const char*>(body.pos.data());
        buffer.insert(buffer.end(), pos_data, pos_data + 3 * sizeof(double));

        const char* vel_data = reinterpret_cast<const char*>(body.vel.data());
        buffer.insert(buffer.end(), vel_data, vel_data + 3 * sizeof(double));
    }

    // Flush buffer to file if it exceeds the limit or at the end of simulation
    if (buffer.size() >= buffer_limit || flush_all) {
        file.write(buffer.data(), buffer.size());
        buffer.clear();
    }

    file.close();
}


// Function to initialize test particles
void initialize_test_particles(std::vector<Body>& bodies, const int num_particles, const double inner_radius, const double outer_radius) {
    const double mass = 1000.0;                    // Mass of test particles (kg)
    const double radius_test_particle = 100.0;                   // Radius of test particles (m)

    for (int i = 0; i < num_particles; ++i) {
        // Random radial position within the ring range
        double radius = std::sqrt((rand() / (double)RAND_MAX)*(outer_radius*outer_radius-inner_radius*inner_radius)+inner_radius*inner_radius);
        
        // Uniform azimuthal distribution
        double angle = 2.0 * M_PI * (rand() / (double)RAND_MAX);
        
        // Calculate position
        std::array<double, 3> pos = {radius * std::cos(angle), radius * std::sin(angle), 0.0};
        
        // Calculate circular velocity with small random perturbation
        double speed = std::sqrt(G * bodies[0].mass / radius); // bodies[0] is Saturn
        double velocity_perturbation = 0.001 * speed * (rand() / (double)RAND_MAX - 0.5); // Small variation
        std::array<double, 3> vel = {-speed * std::sin(angle) + velocity_perturbation,
                                      speed * std::cos(angle) + velocity_perturbation,
                                      0.0};

        // Add the particle to the list
        bodies.emplace_back("TestParticle" + std::to_string(i + 1), mass, radius_test_particle, pos, vel, true);
    }
};

// Function to initialize test particles with linear spacing in radius and the same polar angle
void initialize_test_particles_linearly_spaced(
    std::vector<Body>& bodies,
    const int num_particles,
    const double inner_radius,
    const double outer_radius
) {
    const double mass = 1000.0;  // Mass of test particles (kg)
    const double radius_test_particle = 100.0;                   // Radius of test particles (m)

    // Calculate a constant angle for all particles
    const double angle = 2.0 * M_PI * (rand() / (double)RAND_MAX);  // Random initial angle, same for all particles

    // Calculate linear spacing between particles in the radial direction
    const double delta_radius = (outer_radius - inner_radius) / (num_particles - 1);

    for (int i = 0; i < num_particles; ++i) {
        // Linearly spaced radial position within the ring range
        double radius = inner_radius + i * delta_radius;

        // Calculate position (same angle for all, only radius changes)
        std::array<double, 3> pos = { radius * std::cos(angle), radius * std::sin(angle), 0.0 };

        // Calculate circular velocity with small random perturbation
        double speed = std::sqrt(G * bodies[0].mass / radius); // bodies[0] is Saturn
        double velocity_perturbation = 0.001 * speed * (rand() / (double)RAND_MAX - 0.5);  // Small variation
        std::array<double, 3> vel = { -speed * std::sin(angle) + velocity_perturbation,
                                       speed * std::cos(angle) + velocity_perturbation,
                                       0.0 };

        // Add the particle to the list
        bodies.emplace_back("TestParticle" + std::to_string(i + 1), mass, radius_test_particle, pos, vel, true);
    }
}



void initialize_test_particles_pizza_slice(
    std::vector<Body>& bodies,
    const int num_particles,
    const double inner_radius,
    const double outer_radius,
    const double theta_max // New boundary parameter for angular segment
) {
    const double mass = 1000.0;  // Mass of test particles (kg)
    const double radius_test_particle = 100.0; // Radius of test particles (m)

    for (int i = 0; i < num_particles; ++i) {
        // Random radial position within the ring range
        double radius = std::sqrt((rand() / (double)RAND_MAX) * (outer_radius * outer_radius - inner_radius * inner_radius) + inner_radius * inner_radius);

        // Random azimuthal angle, but only within 0 to theta_max
        double angle = theta_max * (rand() / (double)RAND_MAX);

        // Calculate position
        std::array<double, 3> pos = { radius * std::cos(angle), radius * std::sin(angle), 0.0 };

        // Calculate circular velocity with small random perturbation
        double speed = std::sqrt(G * bodies[0].mass / radius); // bodies[0] is the central body (e.g., Saturn)
        double velocity_perturbation = 0.001 * speed * (rand() / (double)RAND_MAX - 0.5); // Small variation
        std::array<double, 3> vel = { -speed * std::sin(angle) + velocity_perturbation,
                                      speed * std::cos(angle) + velocity_perturbation,
                                      0.0 };

        // Add the particle to the list
        bodies.emplace_back("TestParticle" + std::to_string(i + 1), mass, radius_test_particle, pos, vel, true);
    }
}



// Main function
int main() {
    using namespace std::chrono;

    // Initialize celestial bodies
// Initialize celestial bodies
    std::vector<Body> bodies = {
        {"Saturn", 5.6834e+26, 58232000.0, {68618.46957637797, 285370.0248582383, 3752.786915003062}, {-1.3738454583545705, 0.2902118551465069, -0.008910477308102493}},
        {"Mimas", 3.7493e+19, 198800.0, {184864505.09671617, -22002229.98199375, 252038.23939812754}, {1979.9984526957123, 14127.472465468627, -390.65991177552553}},
        {"Enceladus", 1.08022e+20, 252300.0, {-109337051.10663754, -209954264.08201805, 38521.54148779368}, {11266.18730865185, -5833.758164116197, 0.5136630742954739}},
        {"Tethys", 6.17449e+20, 536300.0, {292419774.94378036, -36542791.048774205, -5405766.343303162}, {1417.779455625827, 11260.975576724282, -55.91207266816317}},
        {"Dione", 1.095452e+21, 562500.0, {-91936871.44147101, -366507891.9207864, 161252.48150243418}, {9704.754830259817, -2441.834545992305, 0.6583216781187331}},
        {"Rhea", 2.306518e+21, 764500.0, {392478412.09623444, -351293703.14145064, 3183371.529527086}, {5667.3714234755325, 6317.217775632378, -1.1196204698413879}},
        {"Titan", 1.3452e+23, 2575500.0, {-284096597.79282385, -1152321961.3854105, -5788718.3471023645}, {5569.79009704894, -1353.1634600278494, 30.102542101274842}},
        {"Hyperion", 5.62e+18, 133000.0, {989608229.6200215, -855298130.9386241, 15059360.39782134}, {3858.8760946532684, 4194.285726077126, 84.42973266855768}},
        {"Iapetus", 1.805635e+21, 734500.0, {-934344577.893008, -3275512258.3243, -729482679.823356}, {3138.7718367337015, -963.9178534695308, 589.0593193080116}},
        {"Phoebe", 8.292e+18, 106600.0, {-9776058739.531536, 26987579.840572786, -4975960593.383701}, {-97.37968968651182, 1982.8379420576377, 216.32902752664063}},
        {"Janus", 1.898e+18, 101700.0, {150511036.3377965, -8960791.757566655, -71129.24605742727}, {1053.6894487395984, 15897.535885269757, -45.491366169378786}},
        {"Epimetheus", 5.264e+17, 64900.00000000001, {148047920.88512397, 27779546.354364473, 243017.50552972933}, {-3028.0484378682086, 15667.401829095017, -95.59911517820149}},
        {"Helene", 1.2e+17, 16000.0, {321458611.3377644, -200365315.96929327, -1091647.8701632577}, {5235.821185388531, 8504.406761553784, 22.473288437898212}},
        {"Telesto", 4.1e+16, 16300.0, {177976926.25615004, 235247359.96620446, -1498509.560591031}, {-9046.869818516356, 6851.224586728263, 226.22660471792216}},
        {"Calypso", 4e+16, 15300.0, {120564939.70208403, -268535115.645305, 4788358.151556253}, {10360.149424676692, 4636.184376963794, -232.8782608095552}},
        {"Atlas", 7e+16, 20500.0, {-8667097.541439574, -136972379.875718, 3269.2360834106294}, {16618.55086255672, -1069.4951902386138, 0.9425617182422849}},
        {"Prometheus", 1.6e+17, 68200.0, {120986403.4493972, -68994263.84415229, -13727.865170405343}, {8188.085207154117, 14367.559198268891, -0.7851500610348839}},
        {"Pandora", 1.4e+17, 52200.0, {-101993173.61049816, -98127629.60421838, 21568.937175860665}, {11423.872623213361, -11750.122871379366, 14.304483297745396}},
        {"Pan", 4950000000000000.0, 17200.0, {122913513.40372911, -52191357.29769165, 3752.7869149953417}, {6634.959471215155, 15535.629722192272, -0.008910477308211793}},
        {"Daphnis", 65000000000000.0, 4600.0, {49531121.81783351, 127516120.785015, 598.9936445031099}, {-15575.443843980447, 6055.025074538545, -0.1741227546267461}}
    };


    const unsigned int matrix_rows = 200;
    const unsigned int matrix_cols = 101;

    // Load matrices for ring potential
    auto matrix_z = load_matrix_file(".\\forces_z_500x401.bin", matrix_rows, matrix_cols);
    auto matrix_r = load_matrix_file(".\\forces_r_500x401.bin", matrix_rows, matrix_cols);


    // Define the local variables R and z
    std::vector<double> R(matrix_rows);  // Define R with the size of MATRIX_ROWS
    std::vector<double> z(matrix_cols);  // Define z with the size of MATRIX_COLS

    // Initialize linspace arrays for ring potential
    for (int i = 0; i < matrix_rows; i++) {
        R[i] = 1e6 + i * (5e8-1e6 / static_cast<double>(matrix_rows) - 1); // Ensure floating-point division
    }
    for (int i = 0; i < matrix_cols; i++) {
        z[i] = -100e6 + i * (200e6 / static_cast<double>(matrix_cols) - 1); // Ensure floating-point division
    }

    // Debug code
    std::vector<double> cur_pos = {0.9e8 * std::cos(M_PI / 4), 0.9e8 * std::cos(M_PI / 4), 0.5e8};
    std::array<double, 3> cur_pos_func = { 0.9e8 * std::cos(M_PI / 4), 0.9e8 * std::cos(M_PI / 4), 0.5e8 };
    std::array<double, 3> cur_force_func = ring_acceleration_cartesian(cur_pos_func, matrix_r, matrix_z, matrix_rows,matrix_cols,R,z);
    std::cout << "cur_force_func: (" << cur_force_func[0] << ", " << cur_force_func[1] << ", " << cur_force_func[2] << ")" << std::endl;

    // initalize test particles from main
    const int local_num_particles = 10;
    const double local_inner_radius = 115000 * 1000.0; // Inner boundary of the ring (meters)
    const double local_outer_radius = 140000 * 1000.0; // Outer boundary of the ring (meters)
    initialize_test_particles(bodies, local_num_particles,local_inner_radius, local_outer_radius);

    auto start_time = high_resolution_clock::now();

    // local var
    const int local_moon_count = 20;
    const double local_dt = 600;
    const double J2 = -16290e-6;

    for (int step = 0; step < timesteps; ++step) {
        compute_accelerations(bodies, matrix_r, matrix_z, matrix_rows, matrix_cols, R, z, J2, local_moon_count);
        update_bodies_leapfrog(bodies, matrix_r, matrix_z, matrix_rows, matrix_cols, R, z, J2, local_moon_count, dt);
        handle_collisions(bodies, local_moon_count);
        //handle_collisions_particles(bodies);
        
        if (step < skipped_timesteps) {
            std::cout << "Step: " << step << "/" << timesteps << " not saved." << std::endl;
        }
        

        else if (step % saved_points_modularity == 0 && step >= skipped_timesteps) {
            std::cout << "Step: " << step << "/" << timesteps << " saved." << std::endl;
            save_to_binary("simulation_output13.bin", bodies);
        }
    }

    // Final flush to ensure all data is saved
    save_to_binary("simulation_output13.bin", bodies, true);

    auto end_time = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end_time - start_time).count();
    std::cout << "Simulation completed in " << duration/1000.0 << " seconds." << std::endl;

    return 0;
}
