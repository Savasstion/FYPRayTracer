#ifndef MATHUTILS_H
#define MATHUTILS_H

//  in case you dont wanna use CUDA's built-in math functions or just dont want to bother with libraries

namespace MathUtils
{
    float fi_sqrt( float number );   //  Quake 3's fast inverse square root
    float approx_sqrt(float number);
}


#endif
