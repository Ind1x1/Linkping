/*
Copyright 2025
Linkping Alibaba Cloude Air (alpha) 2025 Leyi Ye
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "../cuda/comm.cuh"

template<typename T>
__global__ void InitDataKernel(T* ptr, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= size) return;
    ptr[idx] = static_cast<T>(idx);
}

template __global__ void InitDataKernel<__half>(__half*, size_t);
template __global__ void InitDataKernel<float>(float*, size_t);
template __global__ void InitDataKernel<double>(double*, size_t);
template __global__ void InitDataKernel<int>(int*, size_t);
template __global__ void InitDataKernel<int64_t>(int64_t*, size_t);


void LinkPingTimer::TimerProfile(const char* op_name, std::function<void()> func, cudaStream_t stream){
    cudaEvent_t start, stop;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));
    CUDACHECK(cudaEventRecord(start, stream));
    func();
    CUDACHECK(cudaEventRecord(stop, stream));
    CUDACHECK(cudaEventSynchronize(stop));
    cudaEventElapsedTime(&elapsed_time, start, stop);

    printf("%s: %f ms\n", op_name, elapsed_time);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}


void AllReduceGetBw(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks) {
    double baseBw = (double)(count * typesize) / 1.0E9 / sec;
  
    *algBw = baseBw;
    double factor = ((double)(2*(nranks - 1)))/((double)nranks);
    *busBw = baseBw * factor;
}

void InitData(void* data_ptr, size_t size, ncclDataType_t type, cudaStream_t stream) {
    size_t threads = 256;
    size_t blocks = (size + threads - 1) / threads;

    switch (type) {
        case ncclFloat32:
            InitDataKernel<float><<<blocks, threads, 0, stream>>>(static_cast<float*>(data_ptr), size);
            break;
        case ncclFloat16:
            InitDataKernel<__half><<<blocks, threads, 0, stream>>>(static_cast<__half*>(data_ptr), size);
            break;
        case ncclFloat64:
            InitDataKernel<double><<<blocks, threads, 0, stream>>>(static_cast<double*>(data_ptr), size);
            break;
        case ncclInt32:
            InitDataKernel<int><<<blocks, threads, 0, stream>>>(static_cast<int*>(data_ptr), size);
            break;
        case ncclInt64:
            InitDataKernel<int64_t><<<blocks, threads, 0, stream>>>(static_cast<int64_t*>(data_ptr), size);
            break;
        default:
            break;
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
}