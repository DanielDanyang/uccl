#include "tcpx_engine.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(p2p, m) {
    m.doc() = "P2P Engine - TCPX-based peer-to-peer transport";

    m.def("get_oob_ip", &get_oob_ip, "Get the OOB IP address");

    // TcpxEndpoint class binding - 保持与 RDMA 版本相同的接口名 "Endpoint"
    py::class_<TcpxEndpoint>(m, "Endpoint")
        .def(py::init<uint32_t, uint32_t>(), "Create a new TCPX Engine instance",
             py::arg("local_gpu_idx"), py::arg("num_cpus"))
        .def("get_metadata", &TcpxEndpoint::get_metadata, "Get engine metadata")
        .def("get_oob_ip", &TcpxEndpoint::get_oob_ip, "Get OOB IP address")
        .def("get_device_count", &TcpxEndpoint::get_device_count, "Get TCPX device count");
}
