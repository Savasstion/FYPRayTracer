@echo off

pushd ..
Walnut\vendor\bin\premake5.exe vs2022
popd

pushd ..
mkdir "FYPRayTracer\RenderedImages"
popd
pause