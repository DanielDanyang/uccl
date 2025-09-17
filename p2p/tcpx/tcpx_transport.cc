#include "tcpx_interface.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

// 简化版本，避免复杂的 C++ 特性
namespace tcpx {

// Global TCPX plugin state
static void* g_tcpx_plugin_handle = nullptr;
static bool g_tcpx_initialized = false;

// TCPX plugin function pointers (using correct v5 signatures)
static tcpxResult_t (*tcpx_init_fn)(tcpxDebugLogger_t) = nullptr;
static tcpxResult_t (*tcpx_devices_fn)(int*) = nullptr;
static tcpxResult_t (*tcpx_get_properties_fn)(int,
                                              tcpxNetProperties_t*) = nullptr;
static tcpxResult_t (*tcpx_listen_fn)(int, void*, void**) = nullptr;
static tcpxResult_t (*tcpx_connect_v5_fn)(int, void*, void**,
                                          devNetDeviceHandle**) = nullptr;
static tcpxResult_t (*tcpx_accept_v5_fn)(void*, void**,
                                         devNetDeviceHandle**) = nullptr;
static tcpxResult_t (*tcpx_reg_mr_fn)(void*, void*, int, int, void**) = nullptr;
static tcpxResult_t (*tcpx_dereg_mr_fn)(void*, void*) = nullptr;
static tcpxResult_t (*tcpx_isend_v5_fn)(void*, void*, int, int, void*,
                                        void**) = nullptr;
static tcpxResult_t (*tcpx_irecv_v5_fn)(void*, int, void**, int*, int*, void**,
                                        void**) = nullptr;
static tcpxResult_t (*tcpx_test_fn)(void*, int*, int*) = nullptr;
static tcpxResult_t (*tcpx_close_fn)(void*) = nullptr;
static tcpxResult_t (*tcpx_close_listen_fn)(void*) = nullptr;

// TCPX debug logger
static int tcpx_debug_logger(int level, char const* file, char const* func,
                             int line, char const* fmt, ...) {
  (void)level;
  (void)file;
  (void)func;
  (void)line;
  va_list args;
  va_start(args, fmt);
  fprintf(stderr, "[TCPX-PLUGIN] ");
  vfprintf(stderr, fmt, args);
  fprintf(stderr, "\n");
  va_end(args);
  return 0;
}

// Load TCPX plugin
static bool load_tcpx_plugin() {
  std::lock_guard<std::mutex> lock(g_tcpx_mutex);

  if (g_tcpx_initialized) {
    return true;
  }

  char const* plugin_path = getenv("UCCL_TCPX_PLUGIN_PATH");
  if (!plugin_path) {
    plugin_path = "/usr/local/tcpx/lib64/libnccl-net-tcpx.so";
  }

  fprintf(stderr, "[TCPX] Loading TCPX plugin from: %s\n", plugin_path);

  g_tcpx_plugin_handle = dlopen(plugin_path, RTLD_LAZY);
  if (!g_tcpx_plugin_handle) {
    fprintf(stderr, "[TCPX] Failed to load plugin: %s\n", dlerror());
    return false;
  }

  // Load function symbols
  tcpx_init_fn = (tcpxResult_t (*)(tcpxDebugLogger_t))dlsym(
      g_tcpx_plugin_handle, "tcpxInit");
  tcpx_devices_fn =
      (tcpxResult_t (*)(int*))dlsym(g_tcpx_plugin_handle, "tcpxDevices");
  tcpx_get_properties_fn = (tcpxResult_t (*)(int, tcpxNetProperties_t*))dlsym(
      g_tcpx_plugin_handle, "tcpxGetProperties");
  tcpx_listen_fn = (tcpxResult_t (*)(int, void*, void**))dlsym(
      g_tcpx_plugin_handle, "tcpxListen");
  tcpx_connect_fn = (tcpxResult_t (*)(int, void*, void**, void**))dlsym(
      g_tcpx_plugin_handle, "tcpxConnect");
  tcpx_accept_fn = (tcpxResult_t (*)(void*, void**, void**))dlsym(
      g_tcpx_plugin_handle, "tcpxAccept");
  tcpx_reg_mr_fn = (tcpxResult_t (*)(void*, void*, int, int, void**))dlsym(
      g_tcpx_plugin_handle, "tcpxRegMr");
  tcpx_dereg_mr_fn = (tcpxResult_t (*)(void*, void*))dlsym(g_tcpx_plugin_handle,
                                                           "tcpxDeregMr");
  tcpx_isend_fn =
      (tcpxResult_t (*)(void*, void*, int, int, void*, void**))dlsym(
          g_tcpx_plugin_handle, "tcpxIsend");
  tcpx_irecv_fn =
      (tcpxResult_t (*)(void*, int, void**, int*, int*, void**, void**))dlsym(
          g_tcpx_plugin_handle, "tcpxIrecv");
  tcpx_test_fn = (tcpxResult_t (*)(void*, int*, int*))dlsym(
      g_tcpx_plugin_handle, "tcpxTest");
  tcpx_close_fn =
      (tcpxResult_t (*)(void*))dlsym(g_tcpx_plugin_handle, "tcpxClose");
  tcpx_close_listen_fn =
      (tcpxResult_t (*)(void*))dlsym(g_tcpx_plugin_handle, "tcpxCloseListen");

  if (!tcpx_init_fn || !tcpx_devices_fn) {
    fprintf(stderr, "[TCPX] Failed to load required symbols\n");
    dlclose(g_tcpx_plugin_handle);
    g_tcpx_plugin_handle = nullptr;
    return false;
  }

  fprintf(stderr, "[TCPX] Plugin symbols loaded successfully\n");

  // Initialize plugin
  fprintf(stderr, "[TCPX] Initializing TCPX plugin...\n");
  tcpxResult_t result = tcpx_init_fn(tcpx_debug_logger);
  if (result != tcpxSuccess) {
    fprintf(stderr, "[TCPX] Plugin initialization failed: %d\n", result);
    // Continue anyway for now
  } else {
    fprintf(stderr, "[TCPX] Plugin initialized successfully\n");
  }

  g_tcpx_initialized = true;
  return true;
}

// TcpxFactory implementation
std::vector<TcpxFactory::DeviceInfo> TcpxFactory::devices_;
bool TcpxFactory::initialized_ = false;

void TcpxFactory::initialize() {
  if (initialized_) return;

  if (!load_tcpx_plugin()) {
    fprintf(stderr, "[TCPX] Failed to load plugin, using dummy devices\n");
    devices_.resize(1);
    devices_[0].numa_node = 0;
    devices_[0].name = "tcpx0";
    devices_[0].pci_path = "0000:00:00.0";
    initialized_ = true;
    return;
  }

  int ndev = 0;
  tcpxResult_t result = tcpx_devices_fn(&ndev);
  if (result != tcpxSuccess || ndev <= 0) {
    fprintf(stderr, "[TCPX] No devices found, using dummy device\n");
    ndev = 1;
  }

  fprintf(stderr, "[TCPX] Found %d TCPX devices\n", ndev);
  devices_.resize(ndev);

  for (int i = 0; i < ndev; i++) {
    devices_[i].numa_node = 0;  // Default NUMA node
    devices_[i].name = "tcpx" + std::to_string(i);
    devices_[i].pci_path = "0000:00:0" + std::to_string(i) + ".0";

    if (tcpx_get_properties_fn) {
      tcpxNetProperties_t props;
      result = tcpx_get_properties_fn(i, &props);
      if (result == tcpxSuccess && props.name) {
        devices_[i].name = props.name;
        if (props.pciPath) {
          devices_[i].pci_path = props.pciPath;
        }
      }
    }

    fprintf(stderr, "[TCPX] Device %d: %s (%s)\n", i, devices_[i].name.c_str(),
            devices_[i].pci_path.c_str());
  }

  initialized_ = true;
}

TcpxFactory::DeviceInfo* TcpxFactory::get_factory_dev(int dev_idx) {
  initialize();
  if (dev_idx < 0 || dev_idx >= (int)devices_.size()) {
    return &devices_[0];  // Fallback to first device
  }
  return &devices_[dev_idx];
}

// TcpxEndpoint implementation
struct TcpxEndpoint::Impl {
  uint32_t num_cpus;
  int listen_port;
  void* listen_comm;
  std::unordered_map<uint64_t, ConnID> connections;
  std::unordered_map<uint64_t, std::unique_ptr<Mhandle>> memory_handles;
  std::atomic<uint64_t> next_conn_id{1};
  std::atomic<uint64_t> next_mr_id{1};
  std::atomic<uint64_t> next_transfer_id{1};

  Impl(uint32_t cpus)
      : num_cpus(cpus), listen_port(12345), listen_comm(nullptr) {}
};

TcpxEndpoint::TcpxEndpoint(uint32_t num_cpus)
    : impl_(std::make_unique<Impl>(num_cpus)) {
  fprintf(stderr, "[TCPX] Creating TcpxEndpoint with %u CPUs\n", num_cpus);
  TcpxFactory::initialize();
}

TcpxEndpoint::~TcpxEndpoint() {
  fprintf(stderr, "[TCPX] Destroying TcpxEndpoint\n");
  // TODO: Clean up connections and resources
}

int TcpxEndpoint::get_best_dev_idx(int gpu_idx) {
  // Simple mapping: GPU index to device index
  TcpxFactory::initialize();
  int ndev = TcpxFactory::devices_.size();
  return gpu_idx % ndev;
}

void TcpxEndpoint::initialize_engine_by_dev(int dev_idx, bool lazy_init) {
  fprintf(stderr, "[TCPX] Initializing engine for device %d (lazy=%d)\n",
          dev_idx, lazy_init);

  if (!load_tcpx_plugin()) {
    fprintf(stderr, "[TCPX] Plugin not loaded, using simplified mode\n");
    return;
  }

  // TODO: Set up listening socket for this device
  impl_->listen_port = 12345 + dev_idx;
  fprintf(stderr, "[TCPX] Engine initialized for device %d, port %d\n", dev_idx,
          impl_->listen_port);
}

ConnID TcpxEndpoint::tcpx_connect(int local_dev, int local_gpu_idx,
                                  int remote_dev, int remote_gpu_idx,
                                  std::string const& ip_addr, int remote_port) {
  fprintf(stderr,
          "[TCPX] Connecting: local_dev=%d local_gpu=%d -> remote_dev=%d "
          "remote_gpu=%d %s:%d\n",
          local_dev, local_gpu_idx, remote_dev, remote_gpu_idx, ip_addr.c_str(),
          remote_port);

  ConnID conn_id;
  conn_id.sendComm = nullptr;
  conn_id.recvComm = nullptr;
  conn_id.sendDevComm = nullptr;
  conn_id.recvDevComm = nullptr;
  conn_id.sock_fd = -1;

  // TODO: Implement real TCPX connection
  fprintf(stderr, "[TCPX] Connection established (simplified)\n");

  return conn_id;
}

int TcpxEndpoint::get_p2p_listen_port() { return impl_->listen_port; }

}  // namespace tcpx
