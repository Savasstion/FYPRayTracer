#include "MathUtils.h"

float MathUtils::fi_sqrt(float number)
{
    long i;
    float x2, y;
    const float threehalfs = 1.5F;

    x2 = number * 0.5F;
    y  = number;
    i  = * ( long * ) &y;                       // evil floating point bit level hacking
    i  = 0x5f3759df - ( i >> 1 );               // what the fuck?
    y  = * ( float * ) &i;
    y  = y * ( threehalfs - ( x2 * y * y ) );   // 1st iteration
    //	y  = y * ( threehalfs - ( x2 * y * y ) );   // 2nd iteration, this can be removed

    return y;
}

float MathUtils::approx_sqrt(float number) //Newton-Raphson square root
{
    //  std::sqrt is faster so use this when std lib cant be used
    
    if (number <= 0.0f) return 0.0f;

    float x = number;
    float approx = number * 0.5f;
    // Two iterations for reasonable accuracy
    approx = 0.5f * (approx + number / approx);
    approx = 0.5f * (approx + number / approx);
    return approx;
}
