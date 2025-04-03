# mcs.nn~ bending for MaxMSP

This fork gives you acces to parameters of the models used inside mcs.nn~ with MaxMSP.

This work is based on the modified codes for PureData of Błażej Kotowski that you can find [here](https://github.com/blazejkotowski/nn_tilde_bending).

Original code goes to Antoine Cailon, Axel Chemla--Romeu-Santos and the Acids Team at Ircam that you can find [here](https://github.com/acids-ircam/nn_tilde).

## Accessible Functions

- **load *model* *method*** : add the possibility to load a model on the fly (buffer and batches will be the ones originally initialized or by default 2048 and 1).

<img src="/assets/Load.png" width="200">   
&nbsp;

- **layers** : recover a list of all the layers of the loaded neural network.

<img src="/assets/Layers.png" width="600">
&nbsp;

- **get_weights *layer_name*** : recover a list of all the parameters of a given layer. Because Max is limited to lists of size 32767, a sampling method is applied for lists that exceed this size. You can choose between three modes of interpolation with the attribute *downsample_mode*. For better compatibility with Max for Live, lists are sent by chunks of 8192 to avoid crashes.

<img src="/assets/Get.png" width="600">
&nbsp;

- **set_weights *layer_name* *list of weights*** : overwrite parameters of a given layen. As get method can reduce the size of a layer, if the function detects a mismatch between the message from Max and the model then it will upsample the list.

<img src="/assets/Set.png" width="600">
&nbsp;

- **reload** : invert changes made on the model

---

# Build Instructions from main 

## macOS

- Download the latest libtorch (CPU) [here](https://pytorch.org/get-started/locally/) and unzip it to a known directory
- Run the following commands:

```bash
git clone https://github.com/LucasBrgt/mcs.nn_tilde_bending_MaxMSP --recursive
cd mcs.nn_tilde_bending_MaxMSP
mkdir build
cd build
cmake ../src/ -DCMAKE_PREFIX_PATH=/path/to/libtorch -DCMAKE_BUILD_TYPE=Release
make
```

## Windows

- Download Libtorch (CPU) and dependencies [here](https://pytorch.org/get-started/locally/) and unzip to a known directory.
- Install Visual Studio and the C++ tools
- Run the following commands:

```bash
git clone https://github.com/LucasBrgt/mcs.nn_tilde_bending_MaxMSP --recurse-submodules
cd mcs.nn_tilde_bending_MaxMSP
mkdir build
cd build
cmake ..\src -A x64 -DCMAKE_PREFIX_PATH="<unzipped libtorch directory>" -DPUREDATA_INCLUDE_DIR="<path-to-pd/src>" -DPUREDATA_BIN_DIR="<path-to-pd/bin>"
cmake --build . --config Release
```

---
