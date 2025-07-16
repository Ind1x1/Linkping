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

template<typename T>
__global__ void linkping_p2p(T *__restrict__ dest, T const *__restrict__ src, size_t num_elems)
{
    size_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
    size_t gridSize = blockDim.x * gridDim.x;

#pragma unroll(5)
    for (size_t i = globalId; i < num_elems; i += gridSize) {
        dest[i] = src[i];
    }
}

template<typename T>
__global__ void linkping_p2p_ll(T *__restrict__ dest, T const *__restrict__ src, size_t num_elems)
{

    constexpr int VECTOR_SIZE = sizeof(T) >= 8 ? 2 : 4; 
    using VecType = typename std::conditional<sizeof(T) >= 8, 
                                             typename std::conditional<sizeof(T) == 8, double2, float4>::type,
                                             float4>::type;
    
    size_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
    size_t gridSize = blockDim.x * gridDim.x;
    size_t vectorized_elems = (num_elems / VECTOR_SIZE) * VECTOR_SIZE;
    size_t i = globalId * VECTOR_SIZE;
    for (; i < vectorized_elems; i += gridSize * VECTOR_SIZE) {
        VecType* vec_dest = reinterpret_cast<VecType*>(dest + i);
        const VecType* vec_src = reinterpret_cast<const VecType*>(src + i);
        *vec_dest = *vec_src;
    }
    for (size_t j = i; j < num_elems; j += gridSize) {
        dest[j] = src[j];
    }
}

template<typename T>
__global__ void linkping_p2p_ll128(T *__restrict__ dest, T const *__restrict__ src, size_t num_elems)
{
    constexpr int VECTOR_SIZE = 128 / sizeof(T);
    using VecType = typename std::conditional<sizeof(T) == 4, float4,
                                             typename std::conditional<sizeof(T) == 8, double2, float4>::type>::type;
    
    size_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
    size_t gridSize = blockDim.x * gridDim.x;
    
    size_t aligned_elems = (num_elems / VECTOR_SIZE) * VECTOR_SIZE;
    size_t i = globalId * VECTOR_SIZE;
    
    for (; i < aligned_elems; i += gridSize * VECTOR_SIZE) {
        VecType* vec_dest = reinterpret_cast<VecType*>(dest + i);
        const VecType* vec_src = reinterpret_cast<const VecType*>(src + i);
        *vec_dest = *vec_src;
    }
    
    for (size_t j = i; j < num_elems; j += gridSize) {
        dest[j] = src[j];
    }
}

template<typename T>
__global__ void linkping_p2p_simple(T *__restrict__ dest, T const *__restrict__ src, size_t num_elems)
{

    size_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
    size_t gridSize = blockDim.x * gridDim.x;

    for (size_t i = globalId; i < num_elems; i += gridSize) {
        dest[i] = src[i];
    }
}

template<typename T>
float Launch_linkpingp2p_ll(T *dest, T const *src, size_t num_elems, cudaStream_t stream) {
    int blockSize = 0;
    int numBlocks = 0;
    CUDACHECK(cudaOccupancyMaxPotentialBlockSize(&numBlocks, &blockSize, linkping_p2p_ll<T>));
    
    cudaEvent_t start, stop;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));
    
    CUDACHECK(cudaEventRecord(start, stream));
    
    linkping_p2p_ll<T><<<numBlocks, blockSize, 0, stream>>>(dest, src, num_elems);
    
    CUDACHECK(cudaEventRecord(stop, stream));
    CUDACHECK(cudaEventSynchronize(stop));
    float elapsed_time = 0.0f;
    CUDACHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
    CUDACHECK(cudaEventDestroy(start));
    CUDACHECK(cudaEventDestroy(stop));
    
    CUDACHECK(cudaStreamSynchronize(stream));
    return elapsed_time;
}

template<typename T>
float Launch_linkpingp2p_ll128(T *dest, T const *src, size_t num_elems, cudaStream_t stream) {
    int blockSize = 0;
    int numBlocks = 0;
    CUDACHECK(cudaOccupancyMaxPotentialBlockSize(&numBlocks, &blockSize, linkping_p2p_ll128<T>));
    cudaEvent_t start, stop;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));
    
    CUDACHECK(cudaEventRecord(start, stream));
    
    linkping_p2p_ll128<T><<<numBlocks, blockSize, 0, stream>>>(dest, src, num_elems);
    
    CUDACHECK(cudaEventRecord(stop, stream));
    CUDACHECK(cudaEventSynchronize(stop));
    float elapsed_time = 0.0f;
    CUDACHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
    CUDACHECK(cudaEventDestroy(start));
    CUDACHECK(cudaEventDestroy(stop));
    
    CUDACHECK(cudaStreamSynchronize(stream));
    return elapsed_time;
}

template<typename T>
float Launch_linkpingp2p_simple(T *dest, T const *src, size_t num_elems, cudaStream_t stream) {
    int blockSize = 0;
    int numBlocks = 0;
    CUDACHECK(cudaOccupancyMaxPotentialBlockSize(&numBlocks, &blockSize, linkping_p2p_simple<T>));
    cudaEvent_t start, stop;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));

    CUDACHECK(cudaEventRecord(start, stream));
    
    linkping_p2p_simple<T><<<numBlocks, blockSize, 0, stream>>>(dest, src, num_elems);

    CUDACHECK(cudaEventRecord(stop, stream));
    CUDACHECK(cudaEventSynchronize(stop));
    float elapsed_time = 0.0f;
    CUDACHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
    CUDACHECK(cudaEventDestroy(start));
    CUDACHECK(cudaEventDestroy(stop));
    
    CUDACHECK(cudaStreamSynchronize(stream));
    return elapsed_time;
}

__global__ void stridingMemcpyKernel(unsigned int totalThreadCount, unsigned long long loopCount, uint4* dst, uint4* src, size_t chunkSizeInElement) {
    unsigned long long from = blockDim.x * blockIdx.x + thr
    uneadIdx.x;signed long long bigChunkSizeInElement = chunkSizeInElement / 12;
    dst += from;
    src += from;
    uint4* dstBigEnd = dst + (bigChunkSizeInElement * 12) * totalThreadCount;
    uint4* dstEnd = dst + chunkSizeInElement * totalThreadCount;

    for (unsigned int i = 0; i < loopCount; i++) {
        uint4* cdst = dst;
        uint4* csrc = src;

        while (cdst < dstBigEnd) {
            uint4 pipe_0 = *csrc; csrc += totalThreadCount;
            uint4 pipe_1 = *csrc; csrc += totalThreadCount;
            uint4 pipe_2 = *csrc; csrc += totalThreadCount;
            uint4 pipe_3 = *csrc; csrc += totalThreadCount;
            uint4 pipe_4 = *csrc; csrc += totalThreadCount;
            uint4 pipe_5 = *csrc; csrc += totalThreadCount;
            uint4 pipe_6 = *csrc; csrc += totalThreadCount;
            uint4 pipe_7 = *csrc; csrc += totalThreadCount;
            uint4 pipe_8 = *csrc; csrc += totalThreadCount;
            uint4 pipe_9 = *csrc; csrc += totalThreadCount;
            uint4 pipe_10 = *csrc; csrc += totalThreadCount;
            uint4 pipe_11 = *csrc; csrc += totalThreadCount;

            *cdst = pipe_0; cdst += totalThreadCount;
            *cdst = pipe_1; cdst += totalThreadCount;
            *cdst = pipe_2; cdst += totalThreadCount;
            *cdst = pipe_3; cdst += totalThreadCount;
            *cdst = pipe_4; cdst += totalThreadCount;
            *cdst = pipe_5; cdst += totalThreadCount;
            *cdst = pipe_6; cdst += totalThreadCount;
            *cdst = pipe_7; cdst += totalThreadCount;
            *cdst = pipe_8; cdst += totalThreadCount;
            *cdst = pipe_9; cdst += totalThreadCount;
            *cdst = pipe_10; cdst += totalThreadCount;
            *cdst = pipe_11; cdst += totalThreadCount;
        }

        while (cdst < dstEnd) {
            *cdst = *csrc; cdst += totalThreadCount; csrc += totalThreadCount;
        }
    }
}

void Launch_stridingMemcpy(uint4* dst, uint4* src, size_t chunkSizeInElement, cudaStream_t stream) {
    const int numBlocks = 24;
    const int blockSize = 512;
    unsigned int totalThreadCount = numBlocks * blockSize;
    unsigned long long loopCount = 10; 

    cudaEvent_t start, stop;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));

    CUDACHECK(cudaEventRecord(start, stream));

    stridingMemcpyKernel<<<numBlocks, blockSize, 0, stream>>>(
        totalThreadCount,
        loopCount,
        dst,
        src,
        chunkSizeInElement
    );

    CUDACHECK(cudaEventRecord(stop, stream));
    CUDACHECK(cudaEventSynchronize(stop));
    float elapsed_time = 0.0f;
    CUDACHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
    CUDACHECK(cudaEventDestroy(start));
    CUDACHECK(cudaEventDestroy(stop));
    CUDACHECK(cudaStreamSynchronize(stream));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("stridingMemcpyKernel launch failed: %s\n", cudaGetErrorString(err));
    }
    return elapsed_time / 10;

}

template float Launch_linkpingp2p_simple<float>(float*, const float*, size_t, cudaStream_t);
template float Launch_linkpingp2p_ll<float>(float*, const float*, size_t, cudaStream_t);
template float Launch_linkpingp2p_ll128<float>(float*, const float*, size_t, cudaStream_t);

template<typename T>
void Launch_linkpingp2p(T *dest, T const *src, size_t num_elems, cudaStream_t stream) {
    int blockSize = 0;
    int numBlocks = 0;
    CUDACHECK(cudaOccupancyMaxPotentialBlockSize(&numBlocks, &blockSize, linkping_p2p<T>));
    linkping_p2p<T><<<numBlocks, blockSize, 0, stream>>>(dest, src, num_elems);
    CUDACHECK(cudaStreamSynchronize(stream));
}

template void Launch_linkpingp2p<float>(float*, const float*, size_t, cudaStream_t);
template void Launch_linkpingp2p<double>(double*, const double*, size_t, cudaStream_t);
template void Launch_linkpingp2p<int>(int*, const int*, size_t, cudaStream_t);
template void Launch_linkpingp2p<int64_t>(int64_t*, const int64_t*, size_t, cudaStream_t);

void LinkPingTimer::P2PProfile(std::function<void()> func, cudaStream_t stream, size_t count, int typesize) {
    cudaEvent_t start, stop;
    float elapsed_time = 0.0f;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));
    CUDACHECK(cudaEventRecord(start, stream));
    func();
    CUDACHECK(cudaEventRecord(stop, stream));
    CUDACHECK(cudaEventSynchronize(stop));
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("P2P: %f ms, count: %zu, typesize: %d\n", elapsed_time, count, typesize);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void LinkPingTimer::TimerProfile(const char* op_name, std::function<void()> func, cudaStream_t stream, 
                                 size_t count, int typesize, int nranks, int rank) {
    cudaEvent_t start, stop;
    float elapsed_time = 0.0f;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));
    CUDACHECK(cudaEventRecord(start, stream));
    func();
    CUDACHECK(cudaEventRecord(stop, stream));
    CUDACHECK(cudaEventSynchronize(stop));
    cudaEventElapsedTime(&elapsed_time, start, stop);
    
    double sec = elapsed_time / 1000.0;  // 转换为秒
    double algBw, busBw;
    AllReduceGetBw(count, typesize, sec, &algBw, &busBw, nranks);
        printf("%s: %f ms, count: %zu, typesize: %d, Algorithm BW: %.2f GB/s, Bus BW: %.2f GB/s\n", 
               op_name, elapsed_time, count, typesize, algBw, busBw);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void LinkPingTimer::Warmup(const char* op_name, std::function<void()> func, cudaStream_t stream, int warmup_iters) {
    for (int i = 0; i < warmup_iters; ++i) {
        func();
        CUDACHECK(cudaStreamSynchronize(stream));
    }
    //printf("[Warmup] %s finished %d iters\n", op_name, warmup_iters);
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

    size_t max_blocks = 65535;
    if (blocks > max_blocks) {
        blocks = max_blocks;
    }

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

