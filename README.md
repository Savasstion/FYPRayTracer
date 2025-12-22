# Final Year Project (Ray Tracing Sampling Techniques Benchmarker)

This project uses [Walnut](https://github.com/TheCherno/Walnut) as a template.

## Overview
This project implements many ray tracing sampling techniques to be benchmarked and performance logged for comparison purposes. This project is done as my Final Year Project for my Bachelor of Computer Science (Honours) in Interactive Software Technology from Tunku Abdul Rahman University of Management and Technology (TAR UMT).

<img width="1945" height="1067" alt="sneakPeekDemo" src="https://github.com/user-attachments/assets/d5b51134-b2b2-46e4-8a5a-e18286dc687b" />
(A ray-traced banana for reference)

## Requirements
- CUDA Toolkit 12.8.1
- NVIDIA Graphics Card (for CUDA)
- Vulkan SDK
- Assimp

## Features
- Represent and render custom 3D meshes.
- Import 3D model file formats into a scene.
- Uses CUDA to do per-pixel shading.
- Uses Bounding Volume Hierarchy (BVH) to accelerate ray-triangle intersection detection in a two-level acceleration structure system (TLAS/BLAS).
- Configure Scene Object transforms in the editor.
- Configure Material properties in the editor.
- Print out rendered output as an image.
- Ray Tracing Algorithm Selection.
- Rendering Benchmark Result Logging.

## Ray Tracing Sampling Techniques
- Brute Force Ray Tracing (Not really a sampling technique, but it's here to be compared with other techniques)
- Uniform Hemisphere Sampling
- Cosine-Weighted Hemisphere Sampling (better for diffuse materials)
- GGX / Trowbridge-Reitz Sampling (better for specular materials)
- BRDF Sampling (combination of the above two techniques for complex materials that have both diffuse and specular properties)
- [Light Source Sampling using Moreau et al., (2019)](https://doi.org/10.2312/hpg.20191191)
- [Next Event Estimation (NEE)](https://www.cg.tuwien.ac.at/sites/default/files/course/4411/attachments/08_next%20event%20estimation.pdf)
- [ReSTIR DI](https://doi.org/10.1145/3386569.3392481)
- [ReSTIR GI](https://doi.org/10.1111/cgf.14378)

## Executable File
[FYPRayTracer.zip](https://github.com/user-attachments/files/24132515/FYPRayTracer.zip)
- Note: For some reason, some machines even with the cuda dll file, it refuses to run. To fix, install CUDA Toolkit 12.8.1

## How to use in IDE
1. Download the project folder
2. Configure the project for Release x86, hook up all include directories and libraries for CUDA, Vulkan, and Assimp.
3. Done!


## References
Moreau, P., Pharr, M., & Clarberg, P. (2019). Dynamic Many-Light Sampling for Real-Time Ray Tracing. High Performance Graphics, 21–26. https://doi.org/10.2312/hpg.20191191

Bitterli, B., Wyman, C., Pharr, M., Shirley, P., Lefohn, A., & Jarosz, W. (2020). Spatiotemporal reservoir resampling for real-time ray tracing with dynamic direct lighting. ACM Transactions on Graphics, 39(4). https://doi.org/10.1145/3386569.3392481

Ouyang, Y., Liu, S., Kettunen, M., Pharr, M., & Pantaleoni, J. (2021). ReSTIR GI: Path Resampling for Real‐Time Path Tracing. Computer Graphics Forum, 40(8), 17–29. https://doi.org/10.1111/cgf.14378
