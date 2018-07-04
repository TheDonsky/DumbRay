DumbRay
-------

DumbRay is a GPU accelerated path tracer built from ground up, using only CUDA toolkit and virtually nothing else.

Currently, the project is still work in progress and in a highly experimental state, prviding only the minimal command line functionality for testing purposes, but is already capable of producing half-decent images with all of the shadows, reflections, unbiased indirect illumination and alike, scaling well enough on multiple (different) GPU-s and CPU-s on a single machine.

To those, willing to test this one out: Here's provided only the source code of the project and nothing else (for portability reasons). Therefore, one willing to compile can download the source, create a Visual Studio CUDA project and include all the files in it. Provided the dinamic compilation and linking mode is turned on, everything should work just fine.

Unix is currently not supported due to the fact, that the source does not include any code to display images, but that will be dealt with quickly, when the first attempts at user-friendly GUI are made. Other than this, there's no platform exclusivity or something like that...


List of external libraries:
  1. Qt (Not yet full integrated, but we're getting there... Eventually...);
  2. LodePNG (for .png image IO; https://github.com/lvandeve/lodepng; may be no longer needed after full Qt integration).
