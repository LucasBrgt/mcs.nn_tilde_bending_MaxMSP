# mcs.nn~ bending for MaxMSP

This fork gives you acces to parameters of the models used inside mcs.nn~ with MaxMSP.

This work is based on the modified codes for PureData of Błażej Kotowski that you can find [here](https://github.com/blazejkotowski/nn_tilde_bending).

Original code goes to Antoine Cailon, Axel Chemla--Romeu-Santos and the Acids Team at Ircam that you can find [here](https://github.com/acids-ircam/nn_tilde).

## Accessible Functions

- **load *model* *method*** : add the possibility to load a model on the fly (buffer and batches will be the ones originally initialized or by default 2048 and 1).

<img src="/assets/Load.png">   
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
