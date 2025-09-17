#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// UCCL TCPX Engine C API
// Compatible with original UCCL engine API but using TCPX transport

typedef struct uccl_engine uccl_engine;

/**
 * Create a new UCCL TCPX engine instance
 * @param local_gpu_idx Local GPU index
 * @param num_cpus Number of CPU cores
 * @return Pointer to engine instance, or NULL on failure
 */
uccl_engine* uccl_engine_create(uint32_t local_gpu_idx, uint32_t num_cpus);

/**
 * Destroy a UCCL TCPX engine instance
 * @param engine Engine instance to destroy
 */
void uccl_engine_destroy(uccl_engine* engine);

/**
 * Get engine metadata for connection establishment
 * @param engine Engine instance
 * @param metadata_out Output buffer for metadata
 * @param metadata_size_out Size of metadata written
 * @return 0 on success, non-zero on failure
 */
int uccl_engine_get_metadata(uccl_engine* engine, uint8_t* metadata_out, size_t* metadata_size_out);

/**
 * Connect to a remote engine
 * @param engine Engine instance
 * @param remote_ip_addr Remote IP address
 * @param remote_gpu_idx Remote GPU index
 * @param remote_port Remote port (-1 for default)
 * @param conn_id_out Output connection ID
 * @return 0 on success, non-zero on failure
 */
int uccl_engine_connect(uccl_engine* engine, const char* remote_ip_addr, 
                       int remote_gpu_idx, int remote_port, uint64_t* conn_id_out);

/**
 * Register memory region for RDMA operations
 * @param engine Engine instance
 * @param data Pointer to memory region
 * @param size Size of memory region
 * @param mr_id_out Output memory region ID
 * @return 0 on success, non-zero on failure
 */
int uccl_engine_reg(uccl_engine* engine, const void* data, size_t size, uint64_t* mr_id_out);

/**
 * Deregister memory region
 * @param engine Engine instance
 * @param mr_id Memory region ID to deregister
 * @return 0 on success, non-zero on failure
 */
int uccl_engine_dereg(uccl_engine* engine, uint64_t mr_id);

/**
 * Write data to remote memory region
 * @param engine Engine instance
 * @param conn_id Connection ID
 * @param local_data Local data pointer
 * @param remote_data Remote data pointer
 * @param size Data size
 * @param local_mr_id Local memory region ID
 * @param remote_mr_id Remote memory region ID
 * @param transfer_id_out Output transfer ID
 * @return 0 on success, non-zero on failure
 */
int uccl_engine_write(uccl_engine* engine, uint64_t conn_id,
                     const void* local_data, void* remote_data, size_t size,
                     uint64_t local_mr_id, uint64_t remote_mr_id, uint64_t* transfer_id_out);

/**
 * Check if a transfer operation has completed
 * @param engine Engine instance
 * @param conn_id Connection ID
 * @param transfer_id Transfer ID
 * @return 1 if completed, 0 if in progress, -1 on error
 */
int uccl_engine_test_transfer(uccl_engine* engine, uint64_t conn_id, uint64_t transfer_id);

/**
 * Wait for a transfer operation to complete
 * @param engine Engine instance
 * @param conn_id Connection ID
 * @param transfer_id Transfer ID
 * @return 0 on success, non-zero on failure
 */
int uccl_engine_wait_transfer(uccl_engine* engine, uint64_t conn_id, uint64_t transfer_id);

/**
 * Get the listening port for P2P connections
 * @param engine Engine instance
 * @return Port number, or -1 on error
 */
int uccl_engine_get_p2p_listen_port(uccl_engine* engine);

#ifdef __cplusplus
}
#endif
