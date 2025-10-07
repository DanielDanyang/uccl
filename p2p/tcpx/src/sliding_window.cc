/**
 * @file sliding_window.cc
 * @brief Implementation of per-channel sliding window manager
 */

#include "sliding_window.h"
#include "tcpx_interface.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <cuda_runtime.h>

SlidingWindow::SlidingWindow(int max_inflight)
    : max_inflight_(max_inflight) {
  pending_reqs_.reserve(max_inflight);
  pending_indices_.reserve(max_inflight);
  events_.reserve(max_inflight);
}

SlidingWindow::~SlidingWindow() {
  // Clean up via clear() to avoid double-destroy
  clear();
}

bool SlidingWindow::is_full() const {
  return static_cast<int>(pending_reqs_.size()) >= max_inflight_;
}

int SlidingWindow::size() const {
  return static_cast<int>(pending_reqs_.size());
}

void SlidingWindow::add_request(void* request, int chunk_idx, cudaEvent_t event) {
  pending_reqs_.push_back(request);
  pending_indices_.push_back(chunk_idx);
  events_.push_back(event);
}

int SlidingWindow::wait_and_release_oldest(void* comm, bool is_recv) {
  if (pending_reqs_.empty()) {
    return 0;  // Nothing to wait for
  }

  void* oldest_req = pending_reqs_.front();
  int oldest_idx = pending_indices_.front();
  cudaEvent_t oldest_event = events_.front();

  if (is_recv) {
    // Server recv path:
    // 1. Poll tcpx_test() until request completes (drives TCPX progress)
    // 2. Wait for GPU kernel (if using kernel mode)
    // 3. Call tcpx_irecv_consumed() to release TCPX slot

    // Step 1: Wait for TCPX request to complete
    // CRITICAL: Must call tcpx_test() to drive TCPX's internal state machine
    // (see TCPX net_tcpx.cc:tcpxTest() which calls tcpxCommProgress())
    int done = 0;
    int received_size = 0;
    while (!done) {
      if (tcpx_test(oldest_req, &done, &received_size) != 0) {
        std::cerr << "[SlidingWindow] tcpx_test failed for recv chunk "
                  << oldest_idx << std::endl;
        return -1;
      }
      // Small sleep to avoid busy-waiting
      if (!done) {
        std::this_thread::sleep_for(std::chrono::microseconds(1));
      }
    }

    // Step 2: Wait for GPU kernel (if applicable)
    if (oldest_event) {
      // Wait for GPU kernel to finish unpacking
      cudaError_t err = cudaEventSynchronize(oldest_event);
      if (err != cudaSuccess) {
        std::cerr << "[SlidingWindow] cudaEventSynchronize failed for chunk "
                  << oldest_idx << ": " << cudaGetErrorString(err) << std::endl;
        return -1;
      }

      // Destroy the event (we're done with it)
      cudaError_t destroy_err = cudaEventDestroy(oldest_event);
      if (destroy_err != cudaSuccess) {
        std::cerr << "[SlidingWindow] cudaEventDestroy failed: "
                  << cudaGetErrorString(destroy_err) << std::endl;
      }
    }

    // Step 3: Release TCPX request slot
    // Now that tcpx_test() returned done=1, we can safely call irecv_consumed
    if (tcpx_irecv_consumed(comm, 1, oldest_req) != 0) {
      std::cerr << "[SlidingWindow] tcpx_irecv_consumed failed for chunk "
                << oldest_idx << std::endl;
      return -1;
    }

  } else {
    // Client send path: poll until send completes
    // tcpx_test() drives TCPX progress and checks completion

    int done = 0;
    int sent_size = 0;
    while (!done) {
      if (tcpx_test(oldest_req, &done, &sent_size) != 0) {
        std::cerr << "[SlidingWindow] tcpx_test failed for send chunk "
                  << oldest_idx << std::endl;
        return -1;
      }
      // Small sleep to avoid busy-waiting
      if (!done) {
        std::this_thread::sleep_for(std::chrono::microseconds(1));
      }
    }
    // TCPX automatically releases the send request slot when done=1
  }

  // Remove from window
  pending_reqs_.erase(pending_reqs_.begin());
  pending_indices_.erase(pending_indices_.begin());
  events_.erase(events_.begin());

  return 0;
}

int SlidingWindow::drain_all(void* comm, bool is_recv) {
  while (!pending_reqs_.empty()) {
    if (wait_and_release_oldest(comm, is_recv) != 0) {
      std::cerr << "[SlidingWindow] Failed to drain request" << std::endl;
      return -1;
    }
  }
  return 0;
}

void SlidingWindow::clear() {
  pending_reqs_.clear();
  pending_indices_.clear();

  // Clean up events without waiting
  for (auto event : events_) {
    if (event) {
      cudaError_t err = cudaEventDestroy(event);
      if (err != cudaSuccess) {
        std::cerr << "[SlidingWindow] cudaEventDestroy failed in clear(): "
                  << cudaGetErrorString(err) << std::endl;
      }
    }
  }
  events_.clear();
}

