#include "engine.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(p2p, m) {
  m.doc() = "P2P Engine - TCPX-based peer-to-peer transport";

  m.def("get_oob_ip", &get_oob_ip, "Get the OOB IP address");

  // Endpoint class binding - 简化版本
  py::class_<Endpoint>(m, "Endpoint")
      .def(py::init<uint32_t, uint32_t>(), "Create a new TCPX Engine instance",
           py::arg("local_gpu_idx"), py::arg("num_cpus"))
      .def(
          "connect",
          [](Endpoint& self, std::string const& remote_ip_addr,
             int remote_gpu_idx, int remote_port) {
            uint64_t conn_id;
            bool success = self.connect(remote_ip_addr, remote_gpu_idx,
                                        remote_port, conn_id);
            return py::make_tuple(success, conn_id);
          },
          "Connect to a remote server via TCPX", py::arg("remote_ip_addr"),
          py::arg("remote_gpu_idx"), py::arg("remote_port") = -1)
      .def(
          "accept",
          [](Endpoint& self) {
            std::string ip_addr;
            int remote_gpu_idx;
            uint64_t conn_id;
            bool success = self.accept(ip_addr, remote_gpu_idx, conn_id);
            return py::make_tuple(success, ip_addr, remote_gpu_idx, conn_id);
          },
          "Accept an incoming connection via TCPX")
      .def(
          "get_metadata",
          [](Endpoint& self) {
            std::vector<uint8_t> metadata = self.get_metadata();
            return py::bytes(reinterpret_cast<char const*>(metadata.data()),
                             metadata.size());
          },
          "Return endpoint metadata as bytes");
}
