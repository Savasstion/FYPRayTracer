-- premake5.lua
workspace "FYPRayTracer"
   architecture "x64"
   configurations { "Debug", "Release", "Dist" }
   startproject "FYPRayTracer"

outputdir = "%{cfg.buildcfg}-%{cfg.system}-%{cfg.architecture}"
include "Walnut/WalnutExternal.lua"

include "FYPRayTracer"