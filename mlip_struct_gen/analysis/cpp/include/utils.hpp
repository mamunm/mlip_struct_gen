#pragma once

#include <string>
#include <vector>
#include <array>
#include <cmath>
#include <stdexcept>
#include <sstream>

namespace mlip_analysis {

// 3D vector/point representation
struct Vec3 {
    double x, y, z;

    Vec3() : x(0.0), y(0.0), z(0.0) {}
    Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}

    Vec3 operator+(const Vec3& other) const {
        return Vec3(x + other.x, y + other.y, z + other.z);
    }

    Vec3 operator-(const Vec3& other) const {
        return Vec3(x - other.x, y - other.y, z - other.z);
    }

    Vec3 operator*(double scalar) const {
        return Vec3(x * scalar, y * scalar, z * scalar);
    }

    double dot(const Vec3& other) const {
        return x * other.x + y * other.y + z * other.z;
    }

    double norm() const {
        return std::sqrt(x*x + y*y + z*z);
    }
};

// Simulation box with periodic boundary conditions
struct Box {
    Vec3 lo;  // Lower bounds
    Vec3 hi;  // Upper bounds
    Vec3 tilt;  // Triclinic tilt factors (xy, xz, yz)
    bool triclinic;

    Box() : lo(), hi(), tilt(), triclinic(false) {}

    // Get box lengths
    Vec3 lengths() const {
        return Vec3(hi.x - lo.x, hi.y - lo.y, hi.z - lo.z);
    }

    // Apply periodic boundary conditions to a distance vector
    Vec3 apply_pbc(const Vec3& dr) const {
        Vec3 result = dr;
        Vec3 box_lengths = lengths();

        // Handle orthogonal box first
        if (!triclinic) {
            result.x -= std::round(result.x / box_lengths.x) * box_lengths.x;
            result.y -= std::round(result.y / box_lengths.y) * box_lengths.y;
            result.z -= std::round(result.z / box_lengths.z) * box_lengths.z;
        } else {
            // Triclinic box handling
            // For simplicity, we'll use the wrapped coordinates approach
            result.x -= std::round(result.x / box_lengths.x) * box_lengths.x;
            result.y -= std::round(result.y / box_lengths.y) * box_lengths.y;
            result.z -= std::round(result.z / box_lengths.z) * box_lengths.z;
        }

        return result;
    }

    // Compute minimum image distance between two points
    double min_distance(const Vec3& pos1, const Vec3& pos2) const {
        Vec3 dr = pos2 - pos1;
        Vec3 dr_pbc = apply_pbc(dr);
        return dr_pbc.norm();
    }
};

// Atom data structure
struct Atom {
    int id;           // Atom ID
    int type;         // Atom type
    std::string element;  // Element symbol
    Vec3 position;    // Position
    Vec3 velocity;    // Velocity (optional)

    Atom() : id(0), type(0), element(""), position(), velocity() {}
    Atom(int id_, int type_, const std::string& elem, const Vec3& pos)
        : id(id_), type(type_), element(elem), position(pos), velocity() {}
};

// Frame/snapshot from trajectory
struct Frame {
    int timestep;
    Box box;
    std::vector<Atom> atoms;

    Frame() : timestep(0), box(), atoms() {}

    // Get atoms of specific type
    std::vector<size_t> get_atom_indices(const std::string& element) const {
        std::vector<size_t> indices;
        for (size_t i = 0; i < atoms.size(); ++i) {
            if (atoms[i].element == element) {
                indices.push_back(i);
            }
        }
        return indices;
    }

    // Get atoms of specific type index
    std::vector<size_t> get_atom_indices_by_type(int type) const {
        std::vector<size_t> indices;
        for (size_t i = 0; i < atoms.size(); ++i) {
            if (atoms[i].type == type) {
                indices.push_back(i);
            }
        }
        return indices;
    }
};

// String utility functions
inline std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\n\r");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\n\r");
    return str.substr(first, last - first + 1);
}

inline std::vector<std::string> split(const std::string& str, char delimiter = ' ') {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(str);
    while (std::getline(tokenStream, token, delimiter)) {
        std::string trimmed = trim(token);
        if (!trimmed.empty()) {
            tokens.push_back(trimmed);
        }
    }
    return tokens;
}

} // namespace mlip_analysis
