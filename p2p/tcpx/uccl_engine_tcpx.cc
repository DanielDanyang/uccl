// TCPX engine implementation bridging uccl_engine.h to the NCCL GPUDirectTCPX plugin.

#include "../uccl_engine.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>

#include <cerrno>
#include <climits>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <chrono>
#include <cstdlib>

#include <dlfcn.h>
#include <sys/socket.h>
#include <unistd.h>

#if defined(__has_include)
#if __has_include(<cuda_runtime_api.h>)
#define UCCL_TCPX_HAS_CUDA_RUNTIME 1
#include <cuda_runtime_api.h>
#else
#define UCCL_TCPX_HAS_CUDA_RUNTIME 0
#endif
#else
#define UCCL_TCPX_HAS_CUDA_RUNTIME 0
#endif

#ifndef NCCL_NET_HANDLE_MAXSIZE
#define NCCL_NET_HANDLE_MAXSIZE 128
#endif
#ifndef NCCL_PTR_HOST
#define NCCL_PTR_HOST 0x1
#endif
#ifndef NCCL_PTR_CUDA
#define NCCL_PTR_CUDA 0x2
#endif

struct ncclNetDeviceHandle_v7;
typedef struct ncclNetDeviceHandle_v7 ncclNetDeviceHandle_v7_t;

typedef struct {
  char const* name;
  int (*init)(int (*logFunction)(int, char const*, char const*, int,
                                 char const*, ...));
  int (*devices)(int* ndev);
  int (*getProperties)(int dev, void* props_v7);
  int (*listen)(int dev, void* handle, void** listenComm);
  int (*connect)(int dev, void* handle, void** sendComm,
                 ncclNetDeviceHandle_v7_t** sendDevComm);
  int (*accept)(void* listenComm, void** recvComm,
                ncclNetDeviceHandle_v7_t** recvDevComm);
  int (*regMr)(void* comm, void* data, int size, int type, void** mhandle);
  int (*regMrDmaBuf)(void* comm, void* data, size_t size, int type,
                     uint64_t offset, int fd, void** mhandle);
  int (*deregMr)(void* comm, void* mhandle);
  int (*isend)(void* sendComm, void* data, int size, int tag, void* mhandle,
               void** request);
  int (*irecv)(void* recvComm, int n, void** data, int* sizes, int* tags,
               void** mhandles, void** request);
  int (*iflush)(void* recvComm, int n, void** data, int* sizes, void** mhandles,
                void** request);
  int (*test)(void* request, int* done, int* sizes);
  int (*closeSend)(void* sendComm);
  int (*closeRecv)(void* recvComm);
  int (*closeListen)(void* listenComm);
  int (*getDeviceMr)(void* comm, void* mhandle, void** dptr_mhandle);
  int (*irecvConsumed)(void* recvComm, int n, void* request);
} ncclNet_v7_t;

static inline ncclNet_v7_t* resolve_plugin(void* dl) {
  void* sym = dlsym(dl, "ncclNetPlugin_v8");
  if (!sym) sym = dlsym(dl, "ncclNetPlugin_v7");
  return reinterpret_cast<ncclNet_v7_t*>(sym);
}


struct uccl_engine;
struct uccl_conn;
struct uccl_mr;

namespace {

static int detect_ptr_type(void const* ptr) {
#if UCCL_TCPX_HAS_CUDA_RUNTIME
  if (ptr) {
    cudaPointerAttributes attrs{};
    cudaError_t err = cudaPointerGetAttributes(&attrs, ptr);
    if (err == cudaSuccess) {
#if defined(CUDART_VERSION) && CUDART_VERSION >= 10000
      auto memory_type = attrs.type;
#else
      auto memory_type = attrs.memoryType;
#endif
#if defined(cudaMemoryTypeManaged)
      if (memory_type == cudaMemoryTypeDevice ||
          memory_type == cudaMemoryTypeManaged) {
        return NCCL_PTR_CUDA;
      }
#else
      if (memory_type == cudaMemoryTypeDevice) {
        return NCCL_PTR_CUDA;
      }
#endif
      return NCCL_PTR_HOST;
    }
    if (err != cudaErrorInvalidValue) {
      (void)cudaGetLastError();
    }
  }
#endif
  (void)ptr;
  return NCCL_PTR_HOST;
}

struct tcpx_request {
  uccl_conn* conn = nullptr;
  void* request = nullptr;
  void* mhandle = nullptr;
  bool is_send = true;
};

static int send_all(int fd, void const* buf, size_t len) {
  size_t sent = 0;
  auto* bytes = static_cast<unsigned char const*>(buf);
  while (sent < len) {
    ssize_t rc = send(fd, bytes + sent, len - sent, 0);
    if (rc <= 0) return -1;
    sent += static_cast<size_t>(rc);
  }
  return 0;
}

static int recv_all(int fd, void* buf, size_t len) {
  size_t recvd = 0;
  auto* bytes = static_cast<unsigned char*>(buf);
  while (recvd < len) {
    ssize_t rc = recv(fd, bytes + recvd, len - recvd, 0);
    if (rc <= 0) return -1;
    recvd += static_cast<size_t>(rc);
  }
  return 0;
}

}  // namespace

struct uccl_engine {
  void* dl = nullptr;
  ncclNet_v7_t* net = nullptr;
  int dev = 0;
  int oob_listen_fd = -1;
  uint16_t oob_listen_port = 0;
  std::mutex mutex;
};

struct uccl_conn {
  uccl_engine* engine = nullptr;
  int sock_fd = -1;
  void* sendComm = nullptr;
  void* recvComm = nullptr;
};

struct uccl_mr {
  uintptr_t addr = 0;
  size_t size = 0;
  int ptr_type = NCCL_PTR_HOST;
  uccl_engine* engine = nullptr;
};

int uccl_engine_get_fifo_item(uccl_conn_t*, void*) { return -1; }

uccl_engine_t* uccl_engine_create(int /*local_gpu_idx*/, int /*num_cpus*/) {
  auto* eng = new uccl_engine;
  char const* path = getenv("UCCL_TCPX_PLUGIN_PATH");
  if (!path) path = "libnccl-net.so";
  eng->dl = dlopen(path, RTLD_NOW | RTLD_LOCAL);
  if (!eng->dl) {
    delete eng;
    return nullptr;
  }
  eng->net = resolve_plugin(eng->dl);
  if (!eng->net) {
    dlclose(eng->dl);
    delete eng;
    return nullptr;
  }
  if (eng->net->init) eng->net->init(nullptr);
  int ndev = 0;
  if (eng->net->devices) eng->net->devices(&ndev);
  if (ndev <= 0) {
    dlclose(eng->dl);
    delete eng;
    return nullptr;
  }
  char const* dev_env = getenv("UCCL_TCPX_DEV");
  eng->dev = dev_env ? atoi(dev_env) : 0;
  if (eng->dev < 0 || eng->dev >= ndev) eng->dev = 0;
  return eng;
}

void uccl_engine_destroy(uccl_engine_t* engine) {
  if (!engine) return;
  if (engine->oob_listen_fd >= 0) close(engine->oob_listen_fd);
  if (engine->dl) dlclose(engine->dl);
  delete engine;
}

static std::string discover_local_ip() {
  char ipbuf[64] = "127.0.0.1";
  char host[256];
  if (gethostname(host, sizeof(host)) == 0) {
    struct hostent* he = gethostbyname(host);
    if (he && he->h_addrtype == AF_INET && he->h_addr_list[0]) {
      inet_ntop(AF_INET, he->h_addr_list[0], ipbuf, sizeof(ipbuf));
    }
  }
  return std::string(ipbuf);
}

int uccl_engine_get_metadata(uccl_engine_t* engine, char** metadata) {
  if (!engine || !metadata) return -1;
  std::lock_guard<std::mutex> guard(engine->mutex);
  if (engine->oob_listen_fd < 0) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    int yes = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = 0;
    if (bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
      close(fd);
      return -1;
    }
    if (listen(fd, 16) != 0) {
      close(fd);
      return -1;
    }
    socklen_t alen = sizeof(addr);
    if (getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &alen) != 0) {
      close(fd);
      return -1;
    }
    engine->oob_listen_port = ntohs(addr.sin_port);
    engine->oob_listen_fd = fd;
  }

  std::string ip = discover_local_ip();
  std::string payload = ip + ":" + std::to_string(engine->oob_listen_port) + "?0";
  char* out = new char[payload.size() + 1];
  std::strcpy(out, payload.c_str());
  *metadata = out;
  return 0;
}

uccl_conn_t* uccl_engine_accept(uccl_engine_t* engine, char* ip_addr_buf,
                                size_t ip_addr_buf_len, int* remote_gpu_idx) {
  if (!engine || engine->oob_listen_fd < 0 || !ip_addr_buf || !remote_gpu_idx)
    return nullptr;

  struct sockaddr_in peer{};
  socklen_t plen = sizeof(peer);
  int sock = accept(engine->oob_listen_fd, reinterpret_cast<sockaddr*>(&peer),
                    &plen);
  if (sock < 0) return nullptr;

  char ipstr[INET_ADDRSTRLEN] = "";
  inet_ntop(AF_INET, &peer.sin_addr, ipstr, sizeof(ipstr));
  std::strncpy(ip_addr_buf, ipstr, ip_addr_buf_len);
  *remote_gpu_idx = 0;

  unsigned char handle[NCCL_NET_HANDLE_MAXSIZE] = {};
  void* listenComm = nullptr;
  if (engine->net->listen(engine->dev, handle, &listenComm) != 0) {
    close(sock);
    return nullptr;
  }
  if (send_all(sock, handle, NCCL_NET_HANDLE_MAXSIZE) != 0) {
    engine->net->closeListen(listenComm);
    close(sock);
    return nullptr;
  }

  void* recvComm = nullptr;
  ncclNetDeviceHandle_v7_t* devh = nullptr;
  int attempts = 0;
  while (!recvComm && attempts++ < 300) {
    if (engine->net->accept(listenComm, &recvComm, &devh) != 0) {
      engine->net->closeListen(listenComm);
      close(sock);
      return nullptr;
    }
    if (!recvComm) std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  engine->net->closeListen(listenComm);
  if (!recvComm) {
    close(sock);
    return nullptr;
  }

  auto* conn = new uccl_conn;
  conn->engine = engine;
  conn->sock_fd = sock;
  conn->recvComm = recvComm;
  return conn;
}

uccl_conn_t* uccl_engine_connect(uccl_engine_t* engine, char const* ip_addr,
                                 int /*remote_gpu_idx*/, int remote_port) {
  if (!engine || !ip_addr) return nullptr;
  int sock = socket(AF_INET, SOCK_STREAM, 0);
  if (sock < 0) return nullptr;

  struct sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(remote_port);
  if (inet_pton(AF_INET, ip_addr, &addr.sin_addr) != 1) {
    close(sock);
    return nullptr;
  }
  if (connect(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    close(sock);
    return nullptr;
  }

  unsigned char handle[NCCL_NET_HANDLE_MAXSIZE];
  if (recv_all(sock, handle, NCCL_NET_HANDLE_MAXSIZE) != 0) {
    close(sock);
    return nullptr;
  }

  void* sendComm = nullptr;
  ncclNetDeviceHandle_v7_t* devh = nullptr;
  int attempts = 0;
  while (!sendComm && attempts++ < 300) {
    if (engine->net->connect(engine->dev, handle, &sendComm, &devh) != 0) {
      close(sock);
      return nullptr;
    }
    if (!sendComm) std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  if (!sendComm) {
    close(sock);
    return nullptr;
  }

  auto* conn = new uccl_conn;
  conn->engine = engine;
  conn->sock_fd = sock;
  conn->sendComm = sendComm;
  return conn;
}

int uccl_engine_start_listener(uccl_conn_t*) { return 0; }
int uccl_engine_stop_listener(uccl_conn_t*) { return 0; }

uccl_mr_t* uccl_engine_reg(uccl_engine_t* engine, uintptr_t data, size_t size) {
  if (!engine || !data) return nullptr;
  auto* mr = new uccl_mr;
  mr->addr = data;
  mr->size = size;
  mr->ptr_type = detect_ptr_type(reinterpret_cast<void const*>(data));
  mr->engine = engine;
  return mr;
}

static tcpx_request* make_request(uccl_conn* conn, void* request, void* mhandle,
                                  bool is_send) {
  auto* pending = new tcpx_request;
  pending->conn = conn;
  pending->request = request;
  pending->mhandle = mhandle;
  pending->is_send = is_send;
  return pending;
}

int uccl_engine_write(uccl_conn_t* conn, uccl_mr_t* mr, void const* data,
                      size_t size, uint64_t* transfer_id) {
  if (!conn || !conn->engine || !conn->engine->net || !conn->sendComm || !data)
    return -1;
  if (size > static_cast<size_t>(std::numeric_limits<int>::max())) return -1;

  int ptr_type = mr ? mr->ptr_type : detect_ptr_type(data);
  void* mh = nullptr;
  if (conn->engine->net->regMr(conn->sendComm, const_cast<void*>(data),
                               static_cast<int>(size), ptr_type, &mh) != 0) {
    return -1;
  }

  void* request = nullptr;
  if (conn->engine->net->isend(conn->sendComm, const_cast<void*>(data),
                               static_cast<int>(size), 0, mh, &request) != 0 ||
      !request) {
    conn->engine->net->deregMr(conn->sendComm, mh);
    return -1;
  }

  if (!transfer_id) {
    while (true) {
      int done = 0;
      int bytes = 0;
      int rc = conn->engine->net->test(request, &done, &bytes);
      if (rc != 0) done = 1;
      if (done) break;
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    conn->engine->net->deregMr(conn->sendComm, mh);
    return 0;
  }

  *transfer_id = reinterpret_cast<uint64_t>(
      make_request(conn, request, mh, /*is_send=*/true));
  return 0;
}

int uccl_engine_recv(uccl_conn_t* conn, uccl_mr_t* mr, void* data,
                     size_t data_size) {
  if (!conn || !conn->engine || !conn->engine->net || !conn->recvComm || !data)
    return -1;
  if (data_size > static_cast<size_t>(std::numeric_limits<int>::max()))
    return -1;

  int ptr_type = mr ? mr->ptr_type : detect_ptr_type(data);
  void* mh = nullptr;
  if (conn->engine->net->regMr(conn->recvComm, data,
                               static_cast<int>(data_size), ptr_type, &mh) != 0)
    return -1;

  void* request = nullptr;
  void* data_arr[1] = {data};
  int sizes_arr[1] = {static_cast<int>(data_size)};
  int tags_arr[1] = {0};
  void* mhandles_arr[1] = {mh};

  if (conn->engine->net->irecv(conn->recvComm, 1, data_arr, sizes_arr, tags_arr,
                               mhandles_arr, &request) != 0 ||
      !request) {
    conn->engine->net->deregMr(conn->recvComm, mh);
    return -1;
  }

  while (true) {
    int done = 0;
    int bytes = 0;
    int rc = conn->engine->net->test(request, &done, &bytes);
    if (rc != 0) done = 1;
    if (done) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  if (conn->engine->net->irecvConsumed)
    conn->engine->net->irecvConsumed(conn->recvComm, 1, request);

  conn->engine->net->deregMr(conn->recvComm, mh);
  return 0;
}

int uccl_engine_read(uccl_conn_t*, uccl_mr_t*, void const*, size_t, void*,
                     uint64_t*) {
  return -1;  // not supported for TCPX backend
}

bool uccl_engine_xfer_status(uccl_conn_t* conn, uint64_t transfer_id) {
  if (!conn || transfer_id == 0) return true;
  auto* pending = reinterpret_cast<tcpx_request*>(transfer_id);
  int done = 0;
  int bytes = 0;
  int rc = conn->engine->net->test(pending->request, &done, &bytes);
  if (rc != 0) done = 1;
  if (!done) return false;

  if (pending->mhandle) {
    if (pending->is_send)
      conn->engine->net->deregMr(conn->sendComm, pending->mhandle);
    else if (conn->recvComm)
      conn->engine->net->deregMr(conn->recvComm, pending->mhandle);
  }
  delete pending;
  return true;
}

void uccl_engine_conn_destroy(uccl_conn_t* conn) {
  if (!conn) return;
  if (conn->sock_fd >= 0) close(conn->sock_fd);
  if (conn->engine && conn->engine->net) {
    if (conn->sendComm) conn->engine->net->closeSend(conn->sendComm);
    if (conn->recvComm) conn->engine->net->closeRecv(conn->recvComm);
  }
  delete conn;
}

void uccl_engine_mr_destroy(uccl_mr_t* mr) {
  if (mr) delete mr;
}

int uccl_engine_get_sock_fd(uccl_conn_t* conn) {
  return conn ? conn->sock_fd : -1;
}

void uccl_engine_free_endpoint_metadata(uint8_t* metadata) {
  delete[] reinterpret_cast<char*>(metadata);
}
