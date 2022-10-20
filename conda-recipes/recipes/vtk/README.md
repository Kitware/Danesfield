# VTK

## VTK as a dependency

Projects that depend on VTK should add a `yum_requirements.txt` file like `recipe/yum_requirements.txt` in the vtk-feedstock. Without this file, the Linux build fails with `ImportError: No module named vtkRenderingOpenGLPython`.
