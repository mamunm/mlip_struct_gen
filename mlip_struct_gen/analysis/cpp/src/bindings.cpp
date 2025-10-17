#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "../include/utils.hpp"
#include "../include/trajectory_reader.hpp"
#include "../include/rdf_calculator.hpp"

namespace py = pybind11;
using namespace mlip_analysis;

PYBIND11_MODULE(_analysis_core, m) {
    m.doc() = "C++ backend for MLIP trajectory analysis";

    // Vec3 class
    py::class_<Vec3>(m, "Vec3")
        .def(py::init<>())
        .def(py::init<double, double, double>())
        .def_readwrite("x", &Vec3::x)
        .def_readwrite("y", &Vec3::y)
        .def_readwrite("z", &Vec3::z)
        .def("norm", &Vec3::norm)
        .def("dot", &Vec3::dot)
        .def("__repr__", [](const Vec3& v) {
            return "Vec3(" + std::to_string(v.x) + ", " +
                   std::to_string(v.y) + ", " + std::to_string(v.z) + ")";
        });

    // Box class
    py::class_<Box>(m, "Box")
        .def(py::init<>())
        .def_readwrite("lo", &Box::lo)
        .def_readwrite("hi", &Box::hi)
        .def_readwrite("tilt", &Box::tilt)
        .def_readwrite("triclinic", &Box::triclinic)
        .def("lengths", &Box::lengths)
        .def("min_distance", &Box::min_distance)
        .def("apply_pbc", &Box::apply_pbc);

    // Atom class
    py::class_<Atom>(m, "Atom")
        .def(py::init<>())
        .def(py::init<int, int, const std::string&, const Vec3&>())
        .def_readwrite("id", &Atom::id)
        .def_readwrite("type", &Atom::type)
        .def_readwrite("element", &Atom::element)
        .def_readwrite("position", &Atom::position)
        .def_readwrite("velocity", &Atom::velocity);

    // Frame class
    py::class_<Frame>(m, "Frame")
        .def(py::init<>())
        .def_readwrite("timestep", &Frame::timestep)
        .def_readwrite("box", &Frame::box)
        .def_readwrite("atoms", &Frame::atoms)
        .def("get_atom_indices", &Frame::get_atom_indices)
        .def("get_atom_indices_by_type", &Frame::get_atom_indices_by_type)
        .def("__len__", [](const Frame& f) { return f.atoms.size(); });

    // TrajectoryReader class
    py::class_<TrajectoryReader>(m, "TrajectoryReader")
        .def(py::init<const std::string&>())
        .def("set_type_map", &TrajectoryReader::set_type_map)
        .def("read_next_frame", [](TrajectoryReader& reader) {
            Frame frame;
            bool success = reader.read_next_frame(frame);
            if (success) {
                return py::make_tuple(true, frame);
            } else {
                return py::make_tuple(false, frame);
            }
        })
        .def("read_all_frames", &TrajectoryReader::read_all_frames)
        .def("read_frames", &TrajectoryReader::read_frames)
        .def("skip_to_frame", &TrajectoryReader::skip_to_frame)
        .def("count_frames", &TrajectoryReader::count_frames)
        .def("reset", &TrajectoryReader::reset)
        .def("is_open", &TrajectoryReader::is_open);

    // RDFResult class
    py::class_<RDFResult>(m, "RDFResult")
        .def(py::init<>())
        .def_readwrite("r", &RDFResult::r)
        .def_readwrite("gr", &RDFResult::gr)
        .def_readwrite("coordination", &RDFResult::coordination)
        .def_readwrite("pair", &RDFResult::pair)
        .def_readwrite("n_frames", &RDFResult::n_frames)
        .def_readwrite("rmax", &RDFResult::rmax)
        .def_readwrite("nbins", &RDFResult::nbins)
        .def("to_dict", [](const RDFResult& result) {
            py::dict d;
            d["r"] = result.r;
            d["gr"] = result.gr;
            d["coordination"] = result.coordination;
            d["pair"] = result.pair;
            d["n_frames"] = result.n_frames;
            d["rmax"] = result.rmax;
            d["nbins"] = result.nbins;
            return d;
        })
        .def("to_numpy", [](const RDFResult& result) {
            py::dict d;
            // Convert vectors to numpy arrays
            d["r"] = py::array_t<double>(result.r.size(), result.r.data());
            d["gr"] = py::array_t<double>(result.gr.size(), result.gr.data());
            d["coordination"] = py::array_t<double>(result.coordination.size(), result.coordination.data());
            d["pair"] = result.pair;
            d["n_frames"] = result.n_frames;
            d["rmax"] = result.rmax;
            d["nbins"] = result.nbins;
            return d;
        });

    // RDFCalculator class
    py::class_<RDFCalculator>(m, "RDFCalculator")
        .def(py::init<>())
        .def("set_rmax", &RDFCalculator::set_rmax)
        .def("set_nbins", &RDFCalculator::set_nbins)
        .def("get_rmax", &RDFCalculator::get_rmax)
        .def("get_nbins", &RDFCalculator::get_nbins)
        .def("compute_rdf", &RDFCalculator::compute_rdf)
        .def("compute_rdf_single_frame", &RDFCalculator::compute_rdf_single_frame)
        .def("compute_multiple_rdfs", &RDFCalculator::compute_multiple_rdfs)
        .def("compute_multiple_rdfs_dict", [](
            RDFCalculator& calc,
            const std::vector<Frame>& frames,
            const std::vector<std::pair<std::string, std::string>>& pairs
        ) {
            auto results = calc.compute_multiple_rdfs(frames, pairs);
            py::list result_list;
            for (const auto& result : results) {
                result_list.append(result);
            }
            return result_list;
        });

    // Version info
    m.attr("__version__") = "0.1.0";
}
