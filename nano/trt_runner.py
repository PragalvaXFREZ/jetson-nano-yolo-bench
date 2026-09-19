"""
The thinnest possible wrapper around a TensorRT engine. Runs on the Nano only.

The one idea to take away from this file: THE GPU CANNOT SEE NUMPY ARRAYS.
A numpy array lives in "host" memory, owned by the CPU. The engine only reads
and writes "device" memory, owned by the GPU. So every inference is 3 steps:

    1. copy the input   host  -> device     (cudaMemcpy)
    2. run the engine, entirely on the device
    3. copy the output  device -> host      (cudaMemcpy)

Most tutorials use the `pycuda` package for steps 1 and 3. It is not installed
here and takes ~15 minutes to compile, so instead we call NVIDIA's CUDA
runtime library (libcudart.so, already on the Nano) directly through ctypes.
It is the same three C functions pycuda would call for us: cudaMalloc,
cudaMemcpy, cudaFree.

(On the Nano the CPU and GPU physically share the same 4GB of RAM chips, so
these "copies" never leave the chip. The two address spaces are still
separate, which is why the copy is still needed.)

Written for TensorRT 8.0 / Python 3.6, which uses the older "bindings" API.
"""
import ctypes

import numpy as np
import tensorrt as trt

cudart = ctypes.CDLL("libcudart.so")
cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
cudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
cudart.cudaFree.argtypes = [ctypes.c_void_p]
HOST_TO_DEVICE, DEVICE_TO_HOST = 1, 2     # values of the cudaMemcpyKind enum in CUDA's headers


def _check(status, what):
    if status != 0:
        raise RuntimeError("%s failed with CUDA error %d" % (what, status))


class TrtRunner(object):
    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.WARNING)
        # "Deserialize" = load the plan that trtexec spent 15 minutes working
        # out. Takes a few seconds, because no timing has to be redone.
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        # The engine is the read-only plan + weights. The context holds the
        # scratch memory for one run of it.
        self.context = self.engine.create_execution_context()

        # A "binding" is one input or output of the engine. Ours has two:
        # index 0 "images" [1,3,416,416] and index 1 "output0" [1,84,3549].
        # For each one: a numpy array on the host, and a same-sized block on
        # the device. Allocated ONCE here and reused for every frame, because
        # allocating GPU memory per frame would be slow.
        self.host, self.device = [], []
        for i in range(self.engine.num_bindings):
            shape = tuple(self.engine.get_binding_shape(i))
            dtype = trt.nptype(self.engine.get_binding_dtype(i))   # float32 for both, even in an FP16 engine
            arr = np.zeros(shape, dtype=dtype)
            ptr = ctypes.c_void_p()
            _check(cudart.cudaMalloc(ctypes.byref(ptr), arr.nbytes), "cudaMalloc")
            self.host.append(arr)
            self.device.append(ptr)
            if self.engine.binding_is_input(i):
                self.input_index, self.input_shape = i, shape
            else:
                self.output_index = i
        self.size = self.input_shape[2]    # 416

    def infer(self, x):
        """x: [1, 3, S, S] float32, contiguous  ->  [1, 84, N] float32"""
        assert x.shape == self.input_shape and x.dtype == np.float32, (x.shape, x.dtype)
        i, o = self.input_index, self.output_index
        out = self.host[o]

        # 1. host -> device
        _check(cudart.cudaMemcpy(self.device[i], x.ctypes.data, x.nbytes, HOST_TO_DEVICE), "memcpy H2D")
        # 2. run. The engine is handed raw device ADDRESSES, one per binding.
        ok = self.context.execute_v2([int(p.value) for p in self.device])
        if not ok:
            raise RuntimeError("TensorRT execute_v2 failed")
        # 3. device -> host. cudaMemcpy blocks until the GPU is finished, so
        #    when it returns, `out` really holds the result.
        _check(cudart.cudaMemcpy(out.ctypes.data, self.device[o], out.nbytes, DEVICE_TO_HOST), "memcpy D2H")
        return out

    def close(self):
        for p in self.device:
            cudart.cudaFree(p)
        self.device = []
