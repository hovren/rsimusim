# Rolling shutter camera - IMU simulator

**This package is a work in progress and is not supported in anyway!**

This package is a plugin/extension to the IMUSim package which allows to also
simulate a moving rolling shutter camera.

## Simulator workflow
1. Load configuration from `some_config.yml`
1. Load dataset specified in configuration file.
   The dataset is loaded from the current directory, or by using the `datadir=` argument.
1. Create a **new trajectory** which corresponds to the realtive pose between camera and IMU.
1. Simulate IMU data and camera measurements

## Using the simulator results
- Can be loaded by the `SimulationResults` class.
- Only use the trajectory specified in the `SimulationResults` object. 
  **Do not** use the trajectory from the corresponding `Dataset` object since this is not the
   actual trajectory that was used during simulation!
- You can load the landmarks from the `Dataset` object, however.

## Copyright and License
Copyright Hannes Ovrén <hannes.ovren@liu.se>.
Package is released under the GPLv3.
