#define _USE_MATH_DEFINES
#include <cmath>

#include <ctime>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <array>
#include <vector>
#include <string>
#include <chrono>
#include <omp.h> // For parallelization^
#include <unordered_map>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <format>
#include <sstream>

#pragma warning(disable : 4996)

// Constants
const double G = 6.67430e-11; // Gravitational constant (m^3 kg^-1 s^-2)
const double dt = 600;    // Time step (seconds)
const int timesteps = 4000; // Number of time steps to simulate
const double J2 = -16290e-6;  // J2 coefficient (dimensionless)
const double Requa = 6.3781e6; // Equatorial radius of the central body in meters (e.g., Earth)
const double J2_fconstant = J2 * Requa * Requa;
const int num_test_particles = 10000;
const int saved_points_modularity = 100;
const int skipped_timesteps = 2000;
const int moon_count = 20;

const unsigned int MATRIX_ROWS = 200;
const unsigned int MATRIX_COLS = 101;

std::vector<double> R(MATRIX_ROWS), z(MATRIX_COLS); // generated at main

// Structure to represent a celestial body
struct Body {
    std::string name;
    double mass;                     // Mass (kg)
    std::array<double, 3> pos;       // Position {x, y, z} (meters)
    std::array<double, 3> vel;       // Velocity {vx, vy, vz} (m/s)
    std::array<double, 3> acc = {0}; // Acceleration {ax, ay, az} (m/s^2)
    bool is_test_particle = false;   // Flag to indicate if it's a test particle

    Body(const std::string& n, double m, const std::array<double, 3>& p, const std::array<double, 3>& v, bool test_particle = false)
        : name(n), mass(m), pos(p), vel(v), is_test_particle(test_particle) {}
};


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
    const std::vector<std::vector<double>>& matrix_z)
{
    double cur_R = std::sqrt(position[0] * position[0] + position[1] * position[1]);

    // Return empty vector since we are outside the matrix
    if (cur_R < 1e6 || cur_R > 5e8 || position[2] < -100e6 || position[2] > 100e6) {
        return { 0, 0, 0 };
    }

    double calc_dR = (5e8-1e6) / static_cast<double>(MATRIX_ROWS - 1);
    double calc_dz = 200e6 / static_cast<double>(MATRIX_COLS - 1);

    int index_R_first = static_cast<int>(std::floor((cur_R-1e6) / calc_dR));
    int index_R_second = static_cast<int>(std::ceil((cur_R-1e6) / calc_dR));
    int index_z_first = static_cast<int>(std::floor(position[2] / calc_dz)) + (MATRIX_ROWS - 1) / 2;
    int index_z_second = static_cast<int>(std::ceil(position[2] / calc_dz)) + (MATRIX_ROWS - 1) / 2;

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
    const std::vector<std::vector<double>>& matrix_z)
{
    #pragma omp parallel for
    for (size_t i = 0; i < bodies.size(); ++i) {
        if (bodies[i].mass<1) continue;
        
        // Reset acceleration for body i
        // in essence we do acc={0,0,0}; acc += ring_acc so we can just do acc = ring_acc
        // bodies[i].acc = { 0, 0, 0 };

        std::array<double, 3> center = { bodies[i].pos[0] - bodies[0].pos[0],
                                          bodies[i].pos[1] - bodies[0].pos[1],
                                          bodies[i].pos[2] - bodies[0].pos[2] };

        bodies[i].acc = ring_acceleration_cartesian(center, matrix_r, matrix_z);

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


void handle_collisions(std::vector<Body>& bodies) {
    const double moon_radius = 1e6; // Approximate radius for moons (adjust as needed)
    const double out_of_sight_distance = 1e12; // Distance to move particles out of sight
    for (size_t i = moon_count; i < bodies.size(); ++i) {
        if (bodies[i].mass<1) continue;
        // Check for collisions with moons
        for (size_t j = 0; j < 19; ++j) {
            double dx = bodies[i].pos[0] - bodies[j].pos[0];
            double dy = bodies[i].pos[1] - bodies[j].pos[1];
            double dz = bodies[i].pos[2] - bodies[j].pos[2];
            double distance = std::sqrt(dx * dx + dy * dy + dz * dz);

            if (distance < moon_radius) {
                // Move the particle out of sight
                bodies[i].mass = 0;
                bodies[i].pos = {out_of_sight_distance, out_of_sight_distance, out_of_sight_distance};
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
    const std::vector<std::vector<double>>& matrix_z)
{
    // First step: Update velocities by half a step
    #pragma omp parallel for
    for (size_t i = 0; i < bodies.size(); ++i) {
        for (int k = 0; k < 3; ++k) {
            bodies[i].vel[k] += 0.5 * bodies[i].acc[k] * dt;
        }
    }

    // Second step: Update positions using half-step velocities
    #pragma omp parallel for
    for (size_t i = 0; i < bodies.size(); ++i) {
        for (int k = 0; k < 3; ++k) {
            bodies[i].pos[k] += bodies[i].vel[k] * dt;
        }
    }

    // Recompute accelerations after position updates
    compute_accelerations(bodies, matrix_r, matrix_z);

    // Final step: Update velocities by the second half-step
    #pragma omp parallel for
    for (size_t i = 0; i < bodies.size(); ++i) {
        for (int k = 0; k < 3; ++k) {
            bodies[i].vel[k] += 0.5 * bodies[i].acc[k] * dt;
        }
    }
}


// Optimized function to save positions and velocities to a binary file
void save_to_binary(const std::string& filename, const std::vector<Body>& bodies, bool flush_all = false) {
    static std::vector<char> buffer;
    static const size_t buffer_limit = 1024 * 1024; // 1 MB buffer size
    static std::ofstream file(filename, std::ios::binary | std::ios::trunc);

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

    // Close file if flush_all is true
    if (flush_all) {
        file.close();
    }
}


// Function to initialize test particles
void initialize_test_particles(std::vector<Body>& bodies) {
    const double inner_radius = 115000 * 1000.0; // Inner boundary of the ring (meters)
    const double outer_radius = 140000 * 1000.0; // Outer boundary of the ring (meters)
    const double mass = 1000.0;                    // Mass of test particles (kg)
    const int num_particles = num_test_particles;                // Number of test particles

    for (int i = 0; i < num_particles; ++i) {
        // Random radial position within the ring range
        double radius = inner_radius + (outer_radius - inner_radius) * (rand() / (double)RAND_MAX);
        
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
        bodies.emplace_back("TestParticle" + std::to_string(i + 1), mass, pos, vel, true);
    }
};


// Main function
int main() {
    using namespace std::chrono;

    // Initialize celestial bodies
// Initialize celestial bodies
    std::vector<Body> bodies = { 
        {"Saturn", 5.6834e+26, {68618.46945132129, 285370.0254612695, 3752.786849926826}, {-1.3738454583521067, 0.29021185514628206, -0.008910477307691143}},
        {"Mimas", 3.7493e+19, {184864505.2984949, -22002229.98536286, 252038.37892037246}, {1979.9984526941744, 14127.472465471245, -390.6599117770643}},
        {"Enceladus", 1.08022e+20, {-109337051.45078894, -209954263.71984732, 38521.31296500265}, {11266.187308645485, -5833.758164118414, 0.5136630682711915}},
        {"Tethys", 6.17449e+20, {292419775.1733313, -36542790.995482154, -5405766.499901629}, {1417.7794556282245, 11260.975576725457, -55.912072669387435}},
        {"Dione", 1.095452e+21, {-91936871.52056037, -366507891.67800426, 161252.42848012008}, {9704.754830251704, -2441.834545995832, 0.6583216746030806}},
        {"Rhea", 2.306518e+21, {392478412.35828775, -351293703.28780323, 3183371.4393234383}, {5667.371423473723, 6317.21777563236, -1.1196204718953089}},
        {"Titan", 1.3452e+23, {-284096597.30585146, -1152321963.9482226, -5788718.088939952}, {5569.790097038688, -1353.163460026786, 30.102542099666962}},
        {"Hyperion", 5.62e+18, {989608229.6482636, -855298131.5554738, 15059360.01459567}, {3858.87609465001, 4194.285726075834, 84.42973267107726}},
        {"Iapetus", 1.805635e+21, {-934344575.1174282, -3275512257.19237, -729482678.3403598}, {3138.771836728988, -963.9178534761796, 589.059319303903}},
        {"Phoebe", 8.292e+18, {-9776058748.562746, 26987577.644776907, -4975960597.729964}, {-97.37968968877443, 1982.8379420608514, 216.3290275207576}},
        {"Janus", 1.898e+18, {150511036.49406287, -8960791.677340971, -71129.18512030315}, {1053.6894487425939, 15897.535885268851, -45.49136616388283}},
        {"Epimetheus", 5.264e+17, {148047921.0262914, 27779546.37541781, 243017.53103966822}, {-3028.048437870609, 15667.401829093309, -95.5991151744267}},
        {"Helene", 1.2e+17, {321458611.13259506, -200365315.99337092, -1091647.477450326}, {5235.821185381125, 8504.406761553573, 22.473288439907993}},
        {"Telesto", 4.1e+16, {177976926.41962898, 235247359.77733228, -1498509.4768293283}, {-9046.869818518797, 6851.224586729794, 226.22660472127504}},
        {"Calypso", 4e+16, {120564939.70146307, -268535115.30702806, 4788358.111149101}, {10360.149424672109, 4636.184376958853, -232.87826081529397}},
        {"Atlas", 7e+16, {-8667097.566265963, -136972379.65853086, 3269.2186061314947}, {16618.550862551518, -1069.4951902398082, 0.9425617136758603}},
        {"Prometheus", 1.6e+17, {120986403.44925298, -68994263.84358272, -13727.865238072143}, {8188.08520715828, 14367.559198265286, -0.7851500586915522}},
        {"Pandora", 1.4e+17, {-101993173.90043485, -98127629.66224866, 21568.73757781823}, {11423.872623203282, -11750.122871388048, 14.304483292225044}},
        {"Pan", 4950000000000000.0, {122913513.40358953, -52191357.297122635, 3752.7868499328465}, {6634.959471215993, 15535.629722192549, -0.008910477307599463}},
        {"Daphnis", 65000000000000.0, {49531121.78215806, 127516120.83483982, 598.9566076445376}, {-15575.443843985924, 6055.025074537163, -0.1741227541463149}}
    };

    // Load matrices for ring potential
    auto matrix_z = load_matrix_file(".\\forces_z_200x101.bin", MATRIX_ROWS, MATRIX_COLS);
    auto matrix_r = load_matrix_file(".\\forces_r_200x101.bin", MATRIX_ROWS, MATRIX_COLS);

    // Initialize linspace arrays for ring potential
    for (int i = 0; i < MATRIX_ROWS; i++) {
        R[i] = 1e6 + i * ((5e8-1e6) / static_cast<double>(MATRIX_ROWS - 1)); // Ensure floating-point division
    }
    for (int i = 0; i < MATRIX_COLS; i++) {
        z[i] = -100e6 + i * (200e6 / static_cast<double>(MATRIX_COLS - 1)); // Ensure floating-point division
    }

    // Debug code
    std::vector<double> cur_pos = {0.9e8 * std::cos(M_PI / 4), 0.9e8 * std::cos(M_PI / 4), 0.5e8};
    std::array<double, 3> cur_pos_func = { 0.9e8 * std::cos(M_PI / 4), 0.9e8 * std::cos(M_PI / 4), 0.5e8 };
    std::array<double, 3> cur_force_func = ring_acceleration_cartesian(cur_pos_func, matrix_r, matrix_z);
    std::cout << "cur_force_func: (" << cur_force_func[0] << ", " << cur_force_func[1] << ", " << cur_force_func[2] << ")" << std::endl;

    initialize_test_particles(bodies);

    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << "simulation ";
    ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %H-%M-%S");
    ss << ".bin";
    std::string filename = ss.str();
    auto start_time = high_resolution_clock::now();

    for (int step = 0; step < timesteps; ++step) {
        compute_accelerations(bodies, matrix_r, matrix_z);
        update_bodies_leapfrog(bodies, matrix_r, matrix_z);
        handle_collisions(bodies);
        //handle_collisions_particles(bodies);
        
        if (step < skipped_timesteps) {
            std::cout << "Step: " << step << "/" << timesteps << " not saved." << std::endl;
        }
        

        else if (step % saved_points_modularity == 0 && step >= skipped_timesteps) {
            std::cout << "Step: " << step << "/" << timesteps << " saved." << std::endl;
            save_to_binary(filename, bodies);
        }
    }

    // Final flush to ensure all data is saved
    save_to_binary(filename, bodies, true);

    auto end_time = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end_time - start_time).count();
    std::cout << "Simulation completed in " << duration/1000.0 << " seconds." << std::endl;

    return 0;
}
