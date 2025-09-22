#ifndef SAMPLING_TECHNIQUE_ENUM_H
#define SAMPLING_TECHNIQUE_ENUM_H

enum SamplingTechniqueEnum
{
    BRUTE_FORCE,
    UNIFORM_SAMPLING,
    COSINE_WEIGHTED_SAMPLING,
    GGX_SAMPLING,
    BRDF_SAMPLING,
    LIGHT_SOURCE_SAMPLING,
    NEE,
    RESTIR_DI,
    RESTIR_GI,
    
    SamplingTechniqueEnum_COUNT //  used to determine total num of sampling techniques
};

#endif