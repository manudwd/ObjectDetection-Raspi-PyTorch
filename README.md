# ObjDetection-Raspi-4

Object detection implemented on a Raspberry Pi 4 with Transfer Learning using a Resnet18 model.
This a simple tutorial to implement Custom Object Detection with Raspberry Pi 4 using Custom models and custom data with Pytorch, but also extendable to lighter models such as TFlite and ONNX for faster inference.

# Setting up Raspberry Pi-4

You will need:

- Raspberry Pi 4
- Raspberry Monitor Keyboard and Mouse ( In case you plan to use it with a monitor setup)


- RTJ Cable ( In case you're using a remote setup)
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

* Once all is set up, refer to ```infer.py``` , I used Transfer Learning to train my custom data set which can be found in another Repository ```Invisible_Fence_830``` that contains the data used in this study. However, in this study a pretrained ResNet18 is used to train on the custom data, the weights for which can be found here, however, data can similarly be processed and a custom model can be used in its stead.
