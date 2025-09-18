#!/usr/bin/env python3
"""TCPX local write test using ctypes to call libuccl_tcpx_engine.so directly."""

from __future__ import annotations

import ctypes
import os
import sys
import time
from pathlib import Path
from typing import Tuple

import torch

# Load the TCPX engine library
LIB_NAME = "libuccl_tcpx_engine.so"
LIB_PATH = Path(__file__).resolve().parent / LIB_NAME

if not LIB_PATH.exists():
    print(f"❌ {LIB_NAME} not found. Run 'make' first.")
    sys.exit(1)

lib = ctypes.CDLL(str(LIB_PATH))

# Set up function prototypes
lib.uccl_engine_create.argtypes = [ctypes.c_int, ctypes.c_int]
lib.uccl_engine_create.restype = ctypes.c_void_p

lib.uccl_engine_destroy.argtypes = [ctypes.c_void_p]
lib.uccl_engine_destroy.restype = None

lib.uccl_engine_get_metadata.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char_p)]
lib.uccl_engine_get_metadata.restype = ctypes.c_int

lib.uccl_engine_free_endpoint_metadata.argtypes = [ctypes.POINTER(ctypes.c_uint8)]
lib.uccl_engine_free_endpoint_metadata.restype = None

os.environ.setdefault("UCCL_RCMODE", "1")
os.environ.setdefault("UCCL_TCPX_PLUGIN_PATH", "/usr/local/tcpx/lib64/libnccl-net-tcpx.so")
os.environ.setdefault("UCCL_TCPX_DEV", "0")


def test_basic_engine_creation():
    """Test basic TCPX engine creation and destruction."""
    print("Testing TCPX engine creation...")

    # Test engine creation
    engine = lib.uccl_engine_create(0, 4)
    if not engine:
        print("❌ Engine creation failed")
        return False

    print("✅ Engine created successfully")

    # Test metadata retrieval
    metadata_ptr = ctypes.c_char_p()
    result = lib.uccl_engine_get_metadata(engine, ctypes.byref(metadata_ptr))
    if result != 0:
        print(f"❌ Metadata retrieval failed: {result}")
        lib.uccl_engine_destroy(engine)
        return False

    metadata = ctypes.string_at(metadata_ptr)
    print(f"✅ Metadata retrieved: {metadata}")

    # Free metadata
    lib.uccl_engine_free_endpoint_metadata(ctypes.cast(metadata_ptr, ctypes.POINTER(ctypes.c_uint8)))

    # Destroy engine
    lib.uccl_engine_destroy(engine)
    print("✅ Engine destroyed successfully")

    return True


def main() -> int:
    print(f"Using TCPX plugin: {os.environ.get('UCCL_TCPX_PLUGIN_PATH', 'libnccl-net.so')}")
    print(f"Using TCPX device: {os.environ.get('UCCL_TCPX_DEV', '0')}")

    if test_basic_engine_creation():
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Tests failed!")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted, terminating")
        sys.exit(1)
