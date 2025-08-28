# Final Year Project (Ray Tracing Sampling Techniques Benchmarker) [Work in Progress]

This project uses [Walnut](https://github.com/TheCherno/Walnut) as a template.

## Requirements
- CUDA Toolkit
- NVIDIA Graphics Card (for CUDA)
- Vulkan SDK

## Features
- Represent and render custom 3D meshes.
- Import 3D model file formats into a scene. [WIP]
- Uses CUDA to do per-pixel shading.
- Uses Bounding Volume Hierarchy (BVH) to accelerate ray-triangle intersection detection in a two-level acceleration structure system (TLAS/BLAS).
- Configure Scene Object transforms in editor. [WIP]
- Configure Material properties in editor. [WIP]
- Print out rendered output as an image.
- Ray Tracing Algorithm Selection. [WIP]
- Rendering Benchmark Result Logging. [WIP]

## Ray Tracing Sampling Techniques
- Brute Force Ray Tracing (Not really a sampling technique but it's here to be compared with other techniques)
- Uniform Hemisphere Sampling
- Cosine-Weighted Hemisphere Sampling (better for diffuse materials)
- GGX / Trowbridge-Reitz Sampling (better for specular materials)
- BRDF Sampling (combination of the above two techniques for complex materials that are both have diffuse and specular properties)
- [Light Source Sampling using Moreau et al., (2019)](https://doi.org/10.2312/hpg.20191191) [WIP]
- [Next Event Estimation (NEE)](https://www.cg.tuwien.ac.at/sites/default/files/course/4411/attachments/08_next%20event%20estimation.pdf) [WIP]
- [ReSTIR DI](https://doi.org/10.1145/3386569.3392481) [WIP]
- [ReSTIR GI](https://doi.org/10.1111/cgf.14378) [WIP]
 


## References
Moreau, P., Pharr, M., & Clarberg, P. (2019). Dynamic Many-Light Sampling for Real-Time Ray Tracing. High Performance Graphics, 21–26. https://doi.org/10.2312/hpg.20191191

Bitterli, B., Wyman, C., Pharr, M., Shirley, P., Lefohn, A., & Jarosz, W. (2020). Spatiotemporal reservoir resampling for real-time ray tracing with dynamic direct lighting. ACM Transactions on Graphics, 39(4). https://doi.org/10.1145/3386569.3392481

Ouyang, Y., Liu, S., Kettunen, M., Pharr, M., & Pantaleoni, J. (2021). ReSTIR GI: Path Resampling for Real‐Time Path Tracing. Computer Graphics Forum, 40(8), 17–29. https://doi.org/10.1111/cgf.14378
