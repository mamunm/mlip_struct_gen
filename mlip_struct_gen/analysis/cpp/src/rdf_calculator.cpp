#include "../include/rdf_calculator.hpp"
#include <cmath>
#include <numeric>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace mlip_analysis {

RDFCalculator::RDFCalculator()
    : rmax_(10.0), nbins_(200), dr_(0.05) {
    update_dr();
}

RDFResult RDFCalculator::compute_rdf(
    const std::vector<Frame>& frames,
    const std::string& element1,
    const std::string& element2
) {
    if (frames.empty()) {
        throw std::runtime_error("No frames provided");
    }

    RDFResult result;
    result.pair = element1 + "-" + element2;
    result.rmax = rmax_;
    result.nbins = nbins_;
    result.n_frames = frames.size();

    // Initialize histogram
    std::vector<double> histogram(nbins_, 0.0);

    bool same_species = (element1 == element2);

    // Accumulate histogram over all frames
    size_t total_atoms1 = 0;
    size_t total_atoms2 = 0;
    double avg_volume = 0.0;

    for (const auto& frame : frames) {
        // Get atom indices for this pair
        auto indices1 = frame.get_atom_indices(element1);
        auto indices2 = frame.get_atom_indices(element2);

        if (indices1.empty() || indices2.empty()) {
            continue;
        }

        total_atoms1 += indices1.size();
        total_atoms2 += indices2.size();

        // Accumulate volume
        Vec3 lengths = frame.box.lengths();
        avg_volume += lengths.x * lengths.y * lengths.z;

        // Compute histogram for this frame
        compute_histogram(frame, indices1, indices2, histogram, same_species);
    }

    if (total_atoms1 == 0 || total_atoms2 == 0) {
        throw std::runtime_error("No atoms found for elements: " + element1 + ", " + element2);
    }

    // Average volume over frames
    avg_volume /= frames.size();

    // Compute average number of atoms
    size_t avg_atoms1 = total_atoms1 / frames.size();
    size_t avg_atoms2 = total_atoms2 / frames.size();

    // Normalize histogram to get g(r)
    std::vector<double> gr(nbins_, 0.0);
    normalize_rdf(gr, histogram, avg_atoms1, avg_atoms2, frames[0].box, frames.size(), same_species);

    // Compute r values (bin centers)
    std::vector<double> r(nbins_);
    for (size_t i = 0; i < nbins_; ++i) {
        r[i] = (i + 0.5) * dr_;
    }

    // Compute density
    double density = static_cast<double>(avg_atoms2) / avg_volume;

    // Compute coordination number
    std::vector<double> coordination = compute_coordination_number(r, gr, density);

    result.r = r;
    result.gr = gr;
    result.coordination = coordination;

    return result;
}

RDFResult RDFCalculator::compute_rdf_single_frame(
    const Frame& frame,
    const std::string& element1,
    const std::string& element2
) {
    std::vector<Frame> frames = {frame};
    return compute_rdf(frames, element1, element2);
}

std::vector<RDFResult> RDFCalculator::compute_multiple_rdfs(
    const std::vector<Frame>& frames,
    const std::vector<std::pair<std::string, std::string>>& pairs
) {
    std::vector<RDFResult> results;
    results.reserve(pairs.size());

    for (const auto& pair : pairs) {
        try {
            auto result = compute_rdf(frames, pair.first, pair.second);
            results.push_back(result);
        } catch (const std::exception& e) {
            std::cerr << "Warning: Failed to compute RDF for " << pair.first
                      << "-" << pair.second << ": " << e.what() << std::endl;
        }
    }

    return results;
}

void RDFCalculator::compute_histogram(
    const Frame& frame,
    const std::vector<size_t>& indices1,
    const std::vector<size_t>& indices2,
    std::vector<double>& histogram,
    bool same_species
) {
    // Loop over all pairs
    for (size_t i = 0; i < indices1.size(); ++i) {
        size_t idx1 = indices1[i];
        const Atom& atom1 = frame.atoms[idx1];

        size_t j_start = same_species ? i + 1 : 0;

        for (size_t j = j_start; j < indices2.size(); ++j) {
            size_t idx2 = indices2[j];

            // Skip self-pairs
            if (idx1 == idx2) {
                continue;
            }

            const Atom& atom2 = frame.atoms[idx2];

            // Compute distance with PBC
            double dist = frame.box.min_distance(atom1.position, atom2.position);

            // Add to histogram
            if (dist < rmax_) {
                size_t bin = static_cast<size_t>(dist / dr_);
                if (bin < nbins_) {
                    histogram[bin] += 1.0;
                }
            }
        }
    }
}

void RDFCalculator::normalize_rdf(
    std::vector<double>& gr,
    const std::vector<double>& histogram,
    size_t n_atoms1,
    size_t n_atoms2,
    const Box& box,
    size_t n_frames,
    bool same_species
) {
    // Compute box volume
    Vec3 lengths = box.lengths();
    double volume = lengths.x * lengths.y * lengths.z;

    // Calculate number of pairs (not atoms!)
    double num_pairs;
    if (same_species) {
        // Same element: N(N-1)/2 pairs
        num_pairs = static_cast<double>(n_atoms1 * (n_atoms1 - 1)) / 2.0;
    } else {
        // Different elements: N1 * N2 pairs
        num_pairs = static_cast<double>(n_atoms1 * n_atoms2);
    }

    // Pair density
    double pair_density = num_pairs / volume;

    // Normalize each bin (use LEFT edge of bin, matching reference code)
    for (size_t i = 0; i < nbins_; ++i) {
        double r_left = i * dr_;

        // Avoid division by zero at very small r
        if (r_left < 1e-9) {
            gr[i] = 0.0;
            continue;
        }

        // Shell volume: 4π * r^2 * dr (using left edge)
        double shell_volume = 4.0 * M_PI * r_left * r_left * dr_;

        // Normalization factor
        double norm_factor = pair_density * shell_volume;

        if (norm_factor > 1e-9) {
            gr[i] = histogram[i] / (n_frames * norm_factor);
        } else {
            gr[i] = 0.0;
        }
    }
}

std::vector<double> RDFCalculator::compute_coordination_number(
    const std::vector<double>& r,
    const std::vector<double>& gr,
    double density
) {
    std::vector<double> coordination(r.size(), 0.0);

    for (size_t i = 0; i < r.size(); ++i) {
        double integral = 0.0;

        // Trapezoidal integration
        for (size_t j = 0; j <= i; ++j) {
            if (j == 0) {
                integral += 0.0;
            } else {
                double dr_val = r[j] - r[j-1];
                double avg_gr = (gr[j] + gr[j-1]) / 2.0;
                double avg_r = (r[j] + r[j-1]) / 2.0;
                integral += 4.0 * M_PI * avg_r * avg_r * avg_gr * dr_val;
            }
        }

        coordination[i] = density * integral;
    }

    return coordination;
}

} // namespace mlip_analysis
