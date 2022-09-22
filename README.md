# NNAPI delegate issue on Snapdragon 888

This repo contains scripts and a tool to reproduce the NNAPI delegate issue with models containing Concatenation and Dense nodes. The issue appears on Snapdragon 888 (we could not reproduce the issue on Snapdragon 855 and Snapdragon 865). Here is a summary of our findings:

1. On devices with Snapdragon 888 (tested with Android 12), the NNAPI delegate always crashes when there is a Quantize node right before a Concatenation node. For instance, consider the following model:

![concat888](https://user-images.githubusercontent.com/45400368/187241574-62d9a210-0402-449d-b963-76c2dbfcd834.png)

When we quantize this model (INT8), tflite converter adds two Quantize nodes right after the inputs (right before the Concatenation node):

![int8concat888](https://user-images.githubusercontent.com/45400368/187242864-fbe162b3-935e-4047-9f0d-e1230f0093a7.png)

Our experiments show that on Snapdragon 888, whenever there is a Quantize node before a Concatenation node, NNAPI delegate crashes. For this specific case, we can avoid the crash by adding two Identity nodes after model's inputs. We created several dummy models and it turned out MaxPool2D(1,1) is the identity node that can resolve the issue (other Identity nodes like Relu could not help us):

![maxpool888](https://user-images.githubusercontent.com/45400368/187245863-07dbd727-6d94-4f71-98a3-2065218da028.png)

The above-mentioned workaround only works when there is a Concatenation node after model inputs. However, we faced cases where tflite converter adds a Quantize node before intra model Concatenation nodes:

![quantizeConcat](https://user-images.githubusercontent.com/45400368/187254806-9820d326-e81a-4de6-b570-6b3f9ff41398.png)

In such cases, the only solution that we could find is to edit the FlatBuffer binary of the tflite model and replace the Quantize node with an Identity node (e.g. Relu). However, unfortunately this workaround could affect model accuracy (specially in light weight models).  

2. On devices with Snapdragon 888 (tested with Android 12), the INT8 tflite version of a model in which the kernel size in at least one of the Dense layers is larger than 1024x3920, always crashes with the NNAPI delegate. We tried different parameters for the Dense layers and it turned out the threshold for the kernel size is between 1024x3920 and 1024x4096. It means that a Dense layer with kernel size of 1024x3920 does not crash but if you increase the kernel size somewhere it will crash on Snapdragon 888.

## Building and converting the model
* `model_files` folder contains simple models representing the above-mentioned issues. 
  * You can also use `generate_dummy_model.py` to build the models and use `convert_model.py` to convert them to tflite.

## tflite_inference tool 
We have implemented a small tool to feed an input to our sample INT8 tflite models using the `NNAPI` delegate.

### PREREQUISITES: ###
* Linux host computer
* Connectivity to the target device via adb
* Android NDK, version 22 or later
* CMake 3.18 or later

### BUILD INSTRUCTIONS ###
* Unzip the `tensorflow_lite_cpp_2_9_1_static.zip` file inside the `tflite_inference_tool` folder.
* In a terminal, from `tflite_inference_tool` folder:
```console
$ mkdir build
$ cd build
$ cmake -G "Unix Makefiles"
        -DCMAKE_SYSTEM_NAME=Android 
        -DANDROID_ABI=arm64-v8a 
        -DANDROID_STL=c++_shared 
        -DANDROID_NATIVE_API_LEVEL=27 
        -DCMAKE_VERBOSE_MAKEFILE=ON 
        -DCMAKE_TOOLCHAIN_FILE=<path-to-ndk>/build/cmake/android.toolchain.cmake 
        -DCMAKE_BUILD_TYPE=Release
        -DTensorFlowLite_ROOT=../tensorflow_lite_cpp_2_9_1_static ..
$ make
```
* Here, you must replace <path-to-ndk> with the absolute path of the ndk installed on your computer. If you installed NDK through Android studio, it is typically located at:
    `/home/<username>/Android/Sdk/ndk/<version>/` on Linux

* `tensorflow_lite_cpp_2_9_1_static` is TensorflowFlow Lite library package.
### Run INSTRUCTIONS ###
WARNING: This step will write to your `/data/local/tmp` folder on device. Please make sure existing files in that folder are backed up as needed.

In a terminal, from `tflite_inference_tool` folder:
```console
$ adb push ./build/model_test /data/local/tmp
$ adb push ./model_files /data/local/tmp
```

To run the tool you should enter the name of the model you would like to run. In the following, we have listed the output of tool when running on Snapdragon 888: 

Sample 1:
```console
$ adb shell "cd /data/local/tmp && LD_LIBRARY_PATH=. ./model_test --model=int8_quantize_concat.tflite"

INFO: Created TensorFlow Lite delegate for NNAPI.
INFO: Initialized TensorFlow Lite runtime.
INFO: Replacing 5 node(s) with delegate (TfLiteNnapiDelegate) node, yielding 1 partitions.
ERROR: NN API returned error ANEURALNETWORKS_OP_FAILED at line 4650 while completing NNAPI compilation.

ERROR: Node number 5 (TfLiteNnapiDelegate) failed to prepare.
ERROR: Restored original execution plan after delegate application failure.
Segmentation fault 
```

Sample 2:
```console
$ adb shell "cd /data/local/tmp && LD_LIBRARY_PATH=. ./model_test --model=int8_large_Dense.tflite"

INFO: Created TensorFlow Lite delegate for NNAPI.
INFO: Initialized TensorFlow Lite runtime.
INFO: Replacing 3 node(s) with delegate (TfLiteNnapiDelegate) node, yielding 1 partitions.
ERROR: NN API returned error ANEURALNETWORKS_OP_FAILED at line 4650 while completing NNAPI compilation.

ERROR: Node number 3 (TfLiteNnapiDelegate) failed to prepare.
ERROR: Restored original execution plan after delegate application failure.
Segmentation fault 
```
