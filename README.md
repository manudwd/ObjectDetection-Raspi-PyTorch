# ObjDetection-Raspi-4

Object detection implemented on a Raspberry Pi 4 with Transfer Learning using a Resnet18 model.
This a simple tutorial to implement Custom Object Detection with Raspberry Pi 4 using Custom models and custom data with Pytorch, but also extendable to lighter models such as TFlite and ONNX for faster inference.

# Setting up Raspberry Pi-4

You will need:

- Raspberry Pi 4
- Raspberry Monitor Keyboard and Mouse ( In case you plan to use it with a monitor setup)
-Camera Module

In case you're using Raspi-Headless:
-Raspberry Pi 4
- RTJ Cable 
- PuTTy
- VNC Viewer
- Camera Module
- Raspberry Power cable 

* Using Raspi Imager you can set up your SD card, you will need a 64-bit OS like a Raspian 64-bit Bullseye

* Once logged into Rasp you will need set the date correctly, to have your WiFi working correctly which you can do by using and setting up the time zone, WAN location etc
```
sudo raspi-config
```
you can also set up date directly by:

```
sudo date -s 'YYYY-MM-DD HH:MM:SS'
```

Once that boots and you complete the initial setup you’ll need to edit the /boot/config.txt file to enable the camera.

```
# This enables the extended features such as the camera.
start_x=1

# This needs to be at least 128M for the camera processing, if it's bigger you can just leave it as is.
gpu_mem=128

# You need to commment/remove the existing camera_auto_detect line since this causes issues with OpenCV/V4L2 capture.
#camera_auto_detect=1
```

# Getting started

When your Raspberry Pi 4 is working well:

In the terminal run:

```
sudo apt-get update
sudo apt-get upgrade

pip install torch torchvision torchaudio
pip install cv2 
pip install numpy--upgrade

```

# Choice of Model

* The model choice will depend on your implementation case, expected accuracy and density of the models. Many of the PyTorch models come with quantization.

* For optimal performance we want a model that’s quantized and fused. Quantized means that it does the computation using int8 which is much more performant than the standard float32 math. Fused means that consecutive operations have been fused together into a more performant version where possible. Commonly things like activations (ReLU) can be merged into the layer before (Conv2d) during inference.

The aarch64 version of pytorch requires using the qnnpack engine.
```
import torch
torch.backends.quantized.engine = 'qnnpack'
```

* Similarly, quantized models can be found for Tensorflow if the implementation is in Tensorflow, however, models can be converted accross with ONXX that provides a user friendly method of exporting models across PyTorch and Tensorflow. 
* 
* Once all is set up, refer to ```infer.py``` , I used Transfer Learning to train my custom data set which can be found in another Repository ```Invisible_Fence_830``` that contains the data used in this study. However, in this study a pretrained ResNet18 is used to train on the custom data, the weights for which can be found here, however, data can similarly be processed and a custom model can be used in its stead. 

