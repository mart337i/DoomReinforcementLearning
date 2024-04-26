# DoomReinforcementLearning

## Building from source

Here we describe how to build ViZDoom from source.
If you want to install pre-build ViZDoom wheels for Python, see [Python quick start](./pythonQuickstart.md).


### Dependencies

To build ViZDoom (regardless of the method), you need to install some dependencies in your system first.


#### Linux

To build ViZDoom on Linux, the following dependencies are required:
* CMake 3.12+
* Make
* GCC 6.0+
* Boost libraries 1.54.0+
* Python 3.7+ for Python binding (optional)

Also some of additionally [ZDoom dependencies](http://zdoom.org/wiki/Compile_ZDoom_on_Linux) are needed.

##### apt-based distros (Ubuntu, Debian, Linux Mint, etc.)

To get all dependencies on apt-based Linux (Ubuntu, Debian, Linux Mint, etc.) execute the following commands in the shell (might require root access).
```sh
# All possible ViZDoom dependencies,
# most are optional and required only to support alternative sound and music backends in the engine
# other can replace libraries that are included in the ViZDoom repository
apt install build-essential cmake git libsdl2-dev libboost-all-dev libopenal-dev \
zlib1g-dev libjpeg-dev tar libbz2-dev libgtk2.0-dev libfluidsynth-dev libgme-dev \
timidity libwildmidi-dev unzip

# Only essential ViZDoom dependencies
apt install build-essential cmake git libboost-all-dev libsdl2-dev libopenal-dev

# Python 3 dependencies (alternatively Anaconda 3 installed)
apt install python3-dev python3-pip
# or install Anaconda 3 and add it to PATH
```
