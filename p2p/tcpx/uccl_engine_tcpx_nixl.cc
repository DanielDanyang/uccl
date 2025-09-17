// TCPX NIXL 插件 - 最小实现，直接复制 uccl_engine_tcpx.cc
// 只实现基本的写入功能

#include "../uccl_engine.h"
#include <arpa/inet.h>
#include <netinet/in.h>
#include <atomic>
#include <cstring>
#include <string>
#include <dlfcn.h>
#include <sys/socket.h>
#include <unistd.h>

#ifndef NCCL_PTR_HOST
#define NCCL_PTR_HOST 0x1
#endif
#ifndef NCCL_PTR_CUDA
#define NCCL_PTR_CUDA 0x2
#endif
#ifndef NCCL_NET_HANDLE_MAXSIZE
#define NCCL_NET_HANDLE_MAXSIZE 128
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

struct uccl_engine {
  void* dl = nullptr;
  ncclNet_v7_t* net = nullptr;
  int dev = 0;
  int oob_listen_fd = -1;
  uint16_t oob_listen_port = 0;
};

struct uccl_conn {
  uint64_t conn_id = 1;
  uccl_engine* engine = nullptr;
  int sock_fd = -1;
  void* sendComm = nullptr;
  void* recvComm = nullptr;
};

struct uccl_mr {
  uint64_t mr_id;
  uccl_engine* engine;
};

// 不需要的函数，直接返回错误
int uccl_engine_get_fifo_item(uccl_conn_t*, void*) { return -1; }

// 创建引擎
uccl_engine_t* uccl_engine_create(int /*local_gpu_idx*/, int /*num_cpus*/) {
  uccl_engine_t* eng = new uccl_engine;
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

// 获取元数据
int uccl_engine_get_metadata(uccl_engine_t* engine, char** metadata) {
  if (!engine || !metadata) return -1;
  if (engine->oob_listen_fd < 0) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    int yes = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = 0;
    if (bind(fd, (struct sockaddr*)&addr, sizeof(addr)) != 0) {
      close(fd);
      return -1;
    }
    if (listen(fd, 16) != 0) {
      close(fd);
      return -1;
    }
    socklen_t alen = sizeof(addr);
    getsockname(fd, (struct sockaddr*)&addr, &alen);
    engine->oob_listen_port = ntohs(addr.sin_port);
    engine->oob_listen_fd = fd;
  }
  char ipbuf[64] = "127.0.0.1";
  {
    char host[256];
    if (gethostname(host, sizeof(host)) == 0) {
      struct hostent* he = gethostbyname(host);
      if (he && he->h_addrtype == AF_INET && he->h_addr_list[0]) {
        inet_ntop(AF_INET, he->h_addr_list[0], ipbuf, sizeof(ipbuf));
      }
    }
  }
  std::string s =
      std::string(ipbuf) + ":" + std::to_string(engine->oob_listen_port) + "?0";
  *metadata = new char[s.size() + 1];
  std::strcpy(*metadata, s.c_str());
  return 0;
}

// 连接函数 - 先只实现 connect，不实现 accept
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
  if (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) != 0) {
    close(sock);
    return nullptr;
  }
  unsigned char handle[NCCL_NET_HANDLE_MAXSIZE];
  size_t recvd = 0;
  while (recvd < NCCL_NET_HANDLE_MAXSIZE) {
    ssize_t r = recv(sock, handle + recvd, NCCL_NET_HANDLE_MAXSIZE - recvd, 0);
    if (r <= 0) {
      close(sock);
      return nullptr;
    }
    recvd += (size_t)r;
  }
  void* sendComm = nullptr;
  ncclNetDeviceHandle_v7_t* devh = nullptr;
  int tries = 0;
  while (!sendComm && tries++ < 300) {
    if (engine->net->connect(engine->dev, handle, &sendComm, &devh) != 0) {
      close(sock);
      return nullptr;
    }
    if (!sendComm) usleep(5000);
  }
  if (!sendComm) {
    close(sock);
    return nullptr;
  }
  uccl_conn_t* conn = new uccl_conn;
  conn->engine = engine;
  conn->sock_fd = sock;
  conn->sendComm = sendComm;
  return conn;
}

// 内存注册 - 简单实现
uccl_mr_t* uccl_engine_reg(uccl_engine_t* engine, uintptr_t data,
                           size_t /*size*/) {
  if (!engine || !data) return nullptr;
  uccl_mr_t* mr = new uccl_mr;
  mr->mr_id = (uint64_t)data;
  mr->engine = engine;
  return mr;
}

// 写数据 - 核心功能
int uccl_engine_write(uccl_conn_t* conn, uccl_mr_t* /*mr*/, void const* data,
                      size_t size, uint64_t* transfer_id) {
  if (!conn || !conn->sendComm || !data) return -1;
  void* mh = nullptr;
  if (conn->engine->net->regMr(conn->sendComm, const_cast<void*>(data),
                               (int)size, NCCL_PTR_HOST, &mh) != 0)
    return -1;
  void* req = nullptr;
  if (conn->engine->net->isend(conn->sendComm, const_cast<void*>(data),
                               (int)size, 0, mh, &req) != 0 ||
      !req)
    return -1;
  if (transfer_id) *transfer_id = reinterpret_cast<uint64_t>(req);
  return 0;
}

// 检查传输状态
bool uccl_engine_xfer_status(uccl_conn_t* conn, uint64_t transfer_id) {
  if (!conn) return true;
  void* req = reinterpret_cast<void*>(transfer_id);
  int done = 0, n = 0;
  (void)conn->engine->net->test(req, &done, &n);
  return done;
}

// 清理函数
void uccl_engine_conn_destroy(uccl_conn_t* conn) {
  if (!conn) return;
  if (conn->sock_fd >= 0) close(conn->sock_fd);
  if (conn->engine && conn->engine->net) {
    if (conn->sendComm) conn->engine->net->closeSend(conn->sendComm);
  }
  delete conn;
}

void uccl_engine_mr_destroy(uccl_mr_t* mr) {
  if (mr) delete mr;
}

// 暂时不实现的函数
uccl_conn_t* uccl_engine_accept(uccl_engine_t*, char*, size_t, int*) {
  return nullptr;
}
int uccl_engine_start_listener(uccl_conn_t*) { return 0; }
int uccl_engine_recv(uccl_conn_t*, uccl_mr_t*, void*, size_t) { return -1; }
int uccl_engine_read(uccl_conn_t*, uccl_mr_t*, void const*, size_t, void*,
                     uint64_t*) {
  return -1;
}
int uccl_engine_get_sock_fd(uccl_conn_t* conn) {
  return conn ? conn->sock_fd : -1;
}
void uccl_engine_free_endpoint_metadata(uint8_t*) {}
