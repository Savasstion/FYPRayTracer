#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H
#include <vector>

// Utility to copy memory from CPU to GPU
template<typename T>
inline void CopyVectorToDevice(const std::vector<T>& src, T*& dst, uint32_t& count)
{
    count = static_cast<uint32_t>(src.size());
    if (count > 0)
    {
        cudaMalloc(&dst, count * sizeof(T));
        cudaError_t err = cudaMemcpy(dst, src.data(), count * sizeof(T), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "cudaMemcpy error: " << cudaGetErrorString(err) << std::endl;
        }
    }
    else
    {
        dst = nullptr;
    }
}


#endif