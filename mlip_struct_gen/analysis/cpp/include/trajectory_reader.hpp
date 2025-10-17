#pragma once

#include "utils.hpp"
#include <fstream>
#include <sstream>
#include <map>
#include <memory>

namespace mlip_analysis {

// LAMMPS trajectory reader
class TrajectoryReader {
public:
    // Constructor
    explicit TrajectoryReader(const std::string& filename);

    // Destructor
    ~TrajectoryReader();

    // Set type map (type index -> element symbol)
    void set_type_map(const std::map<int, std::string>& type_map);

    // Read next frame from trajectory
    bool read_next_frame(Frame& frame);

    // Read all frames from trajectory
    std::vector<Frame> read_all_frames();

    // Read specific number of frames
    std::vector<Frame> read_frames(size_t n_frames);

    // Skip to specific frame number
    bool skip_to_frame(size_t frame_number);

    // Get total number of frames (requires full scan)
    size_t count_frames();

    // Reset to beginning of file
    void reset();

    // Check if file is open and valid
    bool is_open() const { return file_.is_open(); }

private:
    std::ifstream file_;
    std::string filename_;
    std::map<int, std::string> type_map_;
    size_t current_frame_;

    // Parse box bounds from LAMMPS dump
    bool parse_box_bounds(Box& box);

    // Parse atoms from LAMMPS dump
    bool parse_atoms(Frame& frame, size_t n_atoms);

    // Detect column format from header
    std::vector<std::string> parse_column_header();
};

} // namespace mlip_analysis
