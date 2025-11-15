#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H
#include <cuda_runtime_api.h>
#include <vector>

// Utility to copy memory from CPU to GPU
template <typename T>
inline void CopyVectorToDevice(const std::vector<T>& src, T*& dst, uint32_t& count)
{
    cudaFree(dst);
    dst = nullptr;

    count = static_cast<uint32_t>(src.size());
    if (count > 0)
    {
        cudaError_t err;

        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            std::cerr << "cuda shit error: " << cudaGetErrorString(err) << std::endl;
        }

        err = cudaMalloc((void**)&dst, count * sizeof(T));
        if (err != cudaSuccess)
        {
            std::cerr << "cudaMalloc error: " << cudaGetErrorString(err) << std::endl;
        }

        size_t memorySize = count * sizeof(T);
        err = cudaMemcpy(dst, src.data(), memorySize, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            std::cerr << "cudaMemcpy error: " << cudaGetErrorString(err) << std::endl;
        }
    }
    else
    {
        dst = nullptr;
    }
}


#endif
