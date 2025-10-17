#pragma once

#include "utils.hpp"
#include <vector>
#include <string>
#include <map>
#include <utility>

namespace mlip_analysis {

// RDF result structure
struct RDFResult {
    std::vector<double> r;          // Distance bins (center of each bin)
    std::vector<double> gr;         // RDF values g(r)
    std::vector<double> coordination; // Running coordination number
    std::string pair;               // Pair type (e.g., "O-O", "Na-Cl")
    size_t n_frames;                // Number of frames used
    double rmax;                    // Maximum distance
    size_t nbins;                   // Number of bins

    RDFResult() : n_frames(0), rmax(0.0), nbins(0) {}
};

// RDF Calculator
class RDFCalculator {
public:
    // Constructor
    RDFCalculator();

    // Set computation parameters
    void set_rmax(double rmax) { rmax_ = rmax; }
    void set_nbins(size_t nbins) { nbins_ = nbins; }

    // Compute RDF for a specific pair type from multiple frames
    RDFResult compute_rdf(
        const std::vector<Frame>& frames,
        const std::string& element1,
        const std::string& element2
    );

    // Compute RDF for a single frame (useful for debugging)
    RDFResult compute_rdf_single_frame(
        const Frame& frame,
        const std::string& element1,
        const std::string& element2
    );

    // Compute multiple RDFs at once
    std::vector<RDFResult> compute_multiple_rdfs(
        const std::vector<Frame>& frames,
        const std::vector<std::pair<std::string, std::string>>& pairs
    );

    // Get current parameters
    double get_rmax() const { return rmax_; }
    size_t get_nbins() const { return nbins_; }

private:
    double rmax_;      // Maximum distance for RDF
    size_t nbins_;     // Number of bins
    double dr_;        // Bin width

    // Update bin width
    void update_dr() { dr_ = rmax_ / nbins_; }

    // Compute RDF histogram for a single frame
    void compute_histogram(
        const Frame& frame,
        const std::vector<size_t>& indices1,
        const std::vector<size_t>& indices2,
        std::vector<double>& histogram,
        bool same_species
    );

    // Normalize histogram to get g(r)
    void normalize_rdf(
        std::vector<double>& gr,
        const std::vector<double>& histogram,
        size_t n_atoms1,
        size_t n_atoms2,
        const Box& box,
        size_t n_frames,
        bool same_species
    );

    // Compute coordination number from g(r)
    std::vector<double> compute_coordination_number(
        const std::vector<double>& r,
        const std::vector<double>& gr,
        double density
    );
};

} // namespace mlip_analysis
