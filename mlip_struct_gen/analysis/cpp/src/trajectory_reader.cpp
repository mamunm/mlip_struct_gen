#include "../include/trajectory_reader.hpp"
#include <iostream>
#include <algorithm>

namespace mlip_analysis {

TrajectoryReader::TrajectoryReader(const std::string& filename)
    : filename_(filename), current_frame_(0) {
    file_.open(filename_, std::ios::in);
    if (!file_.is_open()) {
        throw std::runtime_error("Failed to open trajectory file: " + filename);
    }
}

TrajectoryReader::~TrajectoryReader() {
    if (file_.is_open()) {
        file_.close();
    }
}

void TrajectoryReader::set_type_map(const std::map<int, std::string>& type_map) {
    type_map_ = type_map;
}

bool TrajectoryReader::read_next_frame(Frame& frame) {
    if (!file_.is_open() || file_.eof()) {
        return false;
    }

    std::string line;
    frame = Frame();  // Reset frame

    // Read "ITEM: TIMESTEP"
    if (!std::getline(file_, line)) {
        return false;
    }
    if (line.find("ITEM: TIMESTEP") == std::string::npos) {
        throw std::runtime_error("Expected ITEM: TIMESTEP, got: " + line);
    }

    // Read timestep
    if (!std::getline(file_, line)) {
        return false;
    }
    frame.timestep = std::stoi(trim(line));

    // Read "ITEM: NUMBER OF ATOMS"
    if (!std::getline(file_, line)) {
        return false;
    }
    if (line.find("ITEM: NUMBER OF ATOMS") == std::string::npos) {
        throw std::runtime_error("Expected ITEM: NUMBER OF ATOMS");
    }

    // Read number of atoms
    size_t n_atoms;
    if (!std::getline(file_, line)) {
        return false;
    }
    n_atoms = std::stoul(trim(line));

    // Read box bounds
    if (!parse_box_bounds(frame.box)) {
        return false;
    }

    // Read atoms
    if (!parse_atoms(frame, n_atoms)) {
        return false;
    }

    current_frame_++;
    return true;
}

std::vector<Frame> TrajectoryReader::read_all_frames() {
    std::vector<Frame> frames;
    reset();

    Frame frame;
    while (read_next_frame(frame)) {
        frames.push_back(frame);
    }

    return frames;
}

std::vector<Frame> TrajectoryReader::read_frames(size_t n_frames) {
    std::vector<Frame> frames;
    frames.reserve(n_frames);

    Frame frame;
    for (size_t i = 0; i < n_frames; ++i) {
        if (!read_next_frame(frame)) {
            break;
        }
        frames.push_back(frame);
    }

    return frames;
}

bool TrajectoryReader::skip_to_frame(size_t frame_number) {
    if (frame_number < current_frame_) {
        reset();
    }

    Frame frame;
    while (current_frame_ < frame_number) {
        if (!read_next_frame(frame)) {
            return false;
        }
    }

    return true;
}

size_t TrajectoryReader::count_frames() {
    size_t original_frame = current_frame_;
    reset();

    size_t count = 0;
    Frame frame;
    while (read_next_frame(frame)) {
        count++;
    }

    reset();
    skip_to_frame(original_frame);

    return count;
}

void TrajectoryReader::reset() {
    file_.clear();
    file_.seekg(0, std::ios::beg);
    current_frame_ = 0;
}

bool TrajectoryReader::parse_box_bounds(Box& box) {
    std::string line;

    // Read "ITEM: BOX BOUNDS"
    if (!std::getline(file_, line)) {
        return false;
    }

    // Check for triclinic
    box.triclinic = (line.find("xy xz yz") != std::string::npos);

    // Read x bounds
    if (!std::getline(file_, line)) {
        return false;
    }
    auto tokens = split(line);
    if (tokens.size() < 2) {
        throw std::runtime_error("Invalid box bounds format");
    }
    box.lo.x = std::stod(tokens[0]);
    box.hi.x = std::stod(tokens[1]);
    if (box.triclinic && tokens.size() >= 3) {
        box.tilt.x = std::stod(tokens[2]);  // xy
    }

    // Read y bounds
    if (!std::getline(file_, line)) {
        return false;
    }
    tokens = split(line);
    if (tokens.size() < 2) {
        throw std::runtime_error("Invalid box bounds format");
    }
    box.lo.y = std::stod(tokens[0]);
    box.hi.y = std::stod(tokens[1]);
    if (box.triclinic && tokens.size() >= 3) {
        box.tilt.y = std::stod(tokens[2]);  // xz
    }

    // Read z bounds
    if (!std::getline(file_, line)) {
        return false;
    }
    tokens = split(line);
    if (tokens.size() < 2) {
        throw std::runtime_error("Invalid box bounds format");
    }
    box.lo.z = std::stod(tokens[0]);
    box.hi.z = std::stod(tokens[1]);
    if (box.triclinic && tokens.size() >= 3) {
        box.tilt.z = std::stod(tokens[2]);  // yz
    }

    return true;
}

bool TrajectoryReader::parse_atoms(Frame& frame, size_t n_atoms) {
    std::string line;

    // Read "ITEM: ATOMS ..." and parse column names directly
    if (!std::getline(file_, line)) {
        return false;
    }

    // Parse column names from this line
    std::vector<std::string> columns;
    if (line.find("ITEM: ATOMS") != std::string::npos) {
        std::string cols = line.substr(line.find("ATOMS") + 6);
        columns = split(cols);
    }

    // Find column indices
    int id_col = -1, type_col = -1, x_col = -1, y_col = -1, z_col = -1;
    int element_col = -1;

    for (size_t i = 0; i < columns.size(); ++i) {
        if (columns[i] == "id") id_col = i;
        else if (columns[i] == "type") type_col = i;
        else if (columns[i] == "x" || columns[i] == "xu") x_col = i;
        else if (columns[i] == "y" || columns[i] == "yu") y_col = i;
        else if (columns[i] == "z" || columns[i] == "zu") z_col = i;
        else if (columns[i] == "element") element_col = i;
    }

    // Read atoms
    frame.atoms.reserve(n_atoms);
    for (size_t i = 0; i < n_atoms; ++i) {
        if (!std::getline(file_, line)) {
            return false;
        }

        auto tokens = split(line);
        if (tokens.empty()) {
            i--;  // Skip empty lines
            continue;
        }

        Atom atom;

        if (id_col >= 0 && id_col < static_cast<int>(tokens.size())) {
            atom.id = std::stoi(tokens[id_col]);
        }

        if (type_col >= 0 && type_col < static_cast<int>(tokens.size())) {
            atom.type = std::stoi(tokens[type_col]);
            // Try to get element from type map
            if (type_map_.count(atom.type)) {
                atom.element = type_map_[atom.type];
            }
        }

        if (element_col >= 0 && element_col < static_cast<int>(tokens.size())) {
            atom.element = tokens[element_col];
        }

        if (x_col >= 0 && x_col < static_cast<int>(tokens.size())) {
            atom.position.x = std::stod(tokens[x_col]);
        }
        if (y_col >= 0 && y_col < static_cast<int>(tokens.size())) {
            atom.position.y = std::stod(tokens[y_col]);
        }
        if (z_col >= 0 && z_col < static_cast<int>(tokens.size())) {
            atom.position.z = std::stod(tokens[z_col]);
        }

        frame.atoms.push_back(atom);
    }

    return true;
}

std::vector<std::string> TrajectoryReader::parse_column_header() {
    // We already read the "ITEM: ATOMS ..." line in parse_atoms
    // So we need to back up and re-read it
    std::streampos current_pos = file_.tellg();
    file_.seekg(-100, std::ios::cur);  // Go back a bit

    std::string line;
    std::getline(file_, line);

    // Restore position (approximately)
    file_.seekg(current_pos, std::ios::beg);

    // Parse columns from "ITEM: ATOMS id type x y z ..."
    std::vector<std::string> columns;
    if (line.find("ITEM: ATOMS") != std::string::npos) {
        std::string cols = line.substr(line.find("ATOMS") + 6);
        columns = split(cols);
    }

    return columns;
}

} // namespace mlip_analysis
