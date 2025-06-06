#ifndef RAY_H
#define RAY_H

#include "../BaseClasses/Vector3f.h"

class Ray
{
private:
    Vector3f origin;
    Vector3f direction;
public:
    Ray() = default;
    Ray(const Vector3f& origin, const Vector3f& direction) : origin(origin), direction(direction){}

    const Vector3f& GetOrigin() const  { return origin; }
    const Vector3f& GetDirection() const { return direction; }
    void SetOrigin(const Vector3f& o) { origin = o; }
    void SetDirection(const Vector3f& d) { direction = d; }
    Vector3f at(float t) const {
        return origin + direction * t;
    }
};

#endif
