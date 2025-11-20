#ifndef SCENE_SETTINGS_H
#define SCENE_SETTINGS_H
#include <glm/vec3.hpp>

struct RenderingSettings
{
    bool toAccumulate = true;
    int lightBounces = 1;
    int sampleCount = 1;
    glm::vec3 skyColor{01, 01, 01};
    SamplingTechniqueEnum currentSamplingTechnique = BRUTE_FORCE;
    int lightCandidateCount = 4;
    
    //  ReSTIR Settings
    bool useTemporalReuse = false;
    bool useSpatialReuse = false;
};


#endif
