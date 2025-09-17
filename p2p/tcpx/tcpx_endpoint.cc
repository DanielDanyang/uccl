#include "tcpx_endpoint.h"
#include <arpa/inet.h>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <unordered_map>
#include <dlfcn.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

namespace tcpx {

struct TcpxEndpoint::Impl {
  void* dl_handle = nullptr;
  ncclNet_v7_t* net = nullptr;
  int device_id = 0;
  void* listen_comm = nullptr;
  int listen_port = 0;

  std::atomic<uint64_t> next_conn_id{1};
  std::atomic<uint64_t> next_mr_id{1};
  std::atomic<uint64_t> next_transfer_id{1};

  std::unordered_map<uint64_t, void*> connections;
  std::unordered_map<uint64_t, void*> memory_regions;
  std::unordered_map<uint64_t, void*> transfers;
  std::mutex conn_mutex;
  std::mutex mr_mutex;
  std::mutex transfer_mutex;

  bool load_plugin() {
    char const* path = getenv("UCCL_TCPX_PLUGIN_PATH");
    if (!path) path = "/usr/local/tcpx/lib64/libnccl-net-tcpx.so";

    dl_handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!dl_handle) {
      fprintf(stderr, "[TCPX] Failed to load plugin: %s\n", dlerror());
      return false;
    }

    void* sym = dlsym(dl_handle, "ncclNetPlugin_v7");
    if (!sym) {
      fprintf(stderr, "[TCPX] Failed to find ncclNetPlugin_v7: %s\n",
              dlerror());
      dlclose(dl_handle);
      return false;
    }

    net = reinterpret_cast<ncclNet_v7_t*>(sym);

    // Skip init for now to avoid segfault
    // if (net->init) net->init(nullptr);

    // Query devices
    int ndev = 0;
    if (net->devices) {
      net->devices(&ndev);
    }
    if (ndev <= 0) {
      fprintf(stderr, "[TCPX] Warning: No devices reported, assuming 1\n");
      ndev = 1;
    }

    char const* dev_env = getenv("UCCL_TCPX_DEV");
    device_id = dev_env ? atoi(dev_env) : 0;
    if (device_id >= ndev) device_id = 0;

    return true;
  }

  bool setup_listener() {
    if (!net || !net->listen) return false;

    // Create a simple handle for listening
    char handle[128] = {0};
    snprintf(handle, sizeof(handle), "tcpx://0.0.0.0:0");

    int result = net->listen(device_id, handle, &listen_comm);
    if (result != 0 || !listen_comm) {
      fprintf(stderr, "[TCPX] Failed to setup listener\n");
      return false;
    }

    // For now, use a fixed port (this should be dynamic in real implementation)
    listen_port = 12345;
    return true;
  }
};

TcpxEndpoint::TcpxEndpoint(uint32_t local_gpu_idx, uint32_t num_cpus)
    : impl_(std::make_unique<Impl>()) {
  (void)local_gpu_idx;  // unused for now
  (void)num_cpus;       // unused for now

  if (!impl_->load_plugin()) {
    throw std::runtime_error("Failed to load TCPX plugin");
  }

  if (!impl_->setup_listener()) {
    throw std::runtime_error("Failed to setup TCPX listener");
  }
}

TcpxEndpoint::~TcpxEndpoint() {
  if (impl_->listen_comm && impl_->net && impl_->net->closeListen) {
    impl_->net->closeListen(impl_->listen_comm);
  }

  if (impl_->dl_handle) {
    dlclose(impl_->dl_handle);
  }
}

std::vector<uint8_t> TcpxEndpoint::get_metadata() {
  // Simple metadata format: "ip:port?gpu_idx"
  char hostname[256];
  gethostname(hostname, sizeof(hostname));

  struct hostent* he = gethostbyname(hostname);
  std::string ip = "127.0.0.1";
  if (he && he->h_addr_list[0]) {
    ip = inet_ntoa(*((struct in_addr*)he->h_addr_list[0]));
  }

  std::string metadata = ip + ":" + std::to_string(impl_->listen_port) + "?0";
  return std::vector<uint8_t>(metadata.begin(), metadata.end());
}

bool TcpxEndpoint::connect(std::string ip_addr, int remote_gpu_idx,
                           int remote_port, uint64_t& conn_id) {
  if (!impl_->net || !impl_->net->connect) return false;

  // Create connection handle
  std::string handle = "tcpx://" + ip_addr + ":" + std::to_string(remote_port);

  void* send_comm = nullptr;
  void* send_dev_comm = nullptr;

  int result =
      impl_->net->connect(impl_->device_id, const_cast<char*>(handle.c_str()),
                          &send_comm, &send_dev_comm);

  if (result != 0 || !send_comm) {
    fprintf(stderr, "[TCPX] Connect failed\n");
    return false;
  }

  conn_id = impl_->next_conn_id++;

  std::lock_guard<std::mutex> lock(impl_->conn_mutex);
  impl_->connections[conn_id] = send_comm;

  return true;
}

bool TcpxEndpoint::accept(std::string& ip_addr, int& remote_gpu_idx,
                          uint64_t& conn_id) {
  if (!impl_->net || !impl_->net->accept || !impl_->listen_comm) return false;

  void* recv_comm = nullptr;
  void* recv_dev_comm = nullptr;

  int result =
      impl_->net->accept(impl_->listen_comm, &recv_comm, &recv_dev_comm);

  if (result != 0 || !recv_comm) {
    fprintf(stderr, "[TCPX] Accept failed\n");
    return false;
  }

  conn_id = impl_->next_conn_id++;
  ip_addr = "127.0.0.1";  // Placeholder
  remote_gpu_idx = 0;     // Placeholder

  std::lock_guard<std::mutex> lock(impl_->conn_mutex);
  impl_->connections[conn_id] = recv_comm;

  return true;
}

bool TcpxEndpoint::reg(void const* data, size_t size, uint64_t& mr_id) {
  // For now, just assign an ID without actual registration
  // Real implementation would call net->regMr
  mr_id = impl_->next_mr_id++;

  std::lock_guard<std::mutex> lock(impl_->mr_mutex);
  impl_->memory_regions[mr_id] = const_cast<void*>(data);

  return true;
}

bool TcpxEndpoint::dereg(uint64_t mr_id) {
  std::lock_guard<std::mutex> lock(impl_->mr_mutex);
  impl_->memory_regions.erase(mr_id);
  return true;
}

bool TcpxEndpoint::write(uint64_t conn_id, uint64_t mr_id, void const* src,
                         size_t size, void const* remote_addr_info,
                         bool inside_python) {
  // Simplified write implementation
  // Real implementation would use net->isend
  (void)conn_id;
  (void)mr_id;
  (void)src;
  (void)size;
  (void)remote_addr_info;
  (void)inside_python;

  fprintf(stderr, "[TCPX] Write operation (simplified)\n");
  return true;
}

bool TcpxEndpoint::write_async(uint64_t conn_id, uint64_t mr_id,
                               void const* src, size_t size,
                               void const* remote_addr_info,
                               uint64_t* transfer_id) {
  *transfer_id = impl_->next_transfer_id++;
  return write(conn_id, mr_id, src, size, remote_addr_info, false);
}

bool TcpxEndpoint::recv(uint64_t conn_id, uint64_t mr_id, void* dst,
                        size_t size, bool inside_python) {
  // Simplified recv implementation
  (void)conn_id;
  (void)mr_id;
  (void)dst;
  (void)size;
  (void)inside_python;

  fprintf(stderr, "[TCPX] Recv operation (simplified)\n");
  return true;
}

bool TcpxEndpoint::advertise(uint64_t conn_id, uint64_t mr_id, void* addr,
                             size_t len, char* out_buf) {
  // Create a simple advertisement
  snprintf(out_buf, 64, "tcpx_addr_%p_len_%zu", addr, len);
  return true;
}

bool TcpxEndpoint::test_transfer(uint64_t conn_id, uint64_t transfer_id) {
  // For now, always return true (transfer complete)
  return true;
}

bool TcpxEndpoint::wait_transfer(uint64_t conn_id, uint64_t transfer_id) {
  // For now, just return immediately
  return true;
}

}  // namespace tcpx
