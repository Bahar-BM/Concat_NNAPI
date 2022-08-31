#include "tensorflow/lite/c/c_api.h"    
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"

#include <algorithm>
#include <vector>
#include <random>
#include <iostream>
#include <cassert>
#include "cxxopts.hpp"

void Benchmark(TfLiteInterpreter* interpreter, int iterations = 10)
{
    std::chrono::duration<double, std::milli> duration_total(0);
    for (int i = 0; i < iterations; ++i) {
        auto start = std::chrono::steady_clock::now();
        TfLiteInterpreterInvoke(interpreter);
        auto end = std::chrono::steady_clock::now();
        auto const diff = std::chrono::duration<double, std::milli>(end - start);
        duration_total += diff;
    }
    auto average = duration_total / iterations;
    std::cout << average.count() << "ms";
}

std::vector<float> nnapi_inference_single_input(const char* model_path, std::vector<float> const &randomInput, int outputLength)
{
    tflite::StatefulNnApiDelegate::Options opts;
    opts.accelerator_name = "google-edgetpu";
    opts.accelerator_name = "qti-dsp";

    TfLiteDelegate* nnapiDelegate = new tflite::StatefulNnApiDelegate(opts);

    TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();

    TfLiteInterpreterOptionsAddDelegate(options, nnapiDelegate);

    TfLiteModel* model = TfLiteModelCreateFromFile(model_path);
    TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);

    TfLiteInterpreterAllocateTensors(interpreter);
    auto* inputTensor = TfLiteInterpreterGetInputTensor(interpreter, 0);

    auto status = TfLiteTensorCopyFromBuffer(inputTensor, randomInput.data(), randomInput.size()*sizeof(float));
    assert(status == kTfLiteOk);

    TfLiteInterpreterInvoke(interpreter);

    std::vector<float> output(outputLength);
    auto const* outputTensor = TfLiteInterpreterGetOutputTensor(interpreter, 0);

    status = TfLiteTensorCopyToBuffer(outputTensor, output.data(), output.size()*sizeof(float));
    assert(status == kTfLiteOk);

    std::cout<<"\nThe average elapsed time in nnapi delegate: ";
    Benchmark(interpreter);

    TfLiteInterpreterDelete(interpreter);
    delete reinterpret_cast<tflite::StatefulNnApiDelegate*>(nnapiDelegate);
    TfLiteInterpreterOptionsDelete(options);
    TfLiteModelDelete(model);

    return output;
}

std::vector<float> nnapi_inference_two_inputs(const char* model_path, std::vector<float> const &randomInput_0, 
                                               std::vector<float> const &randomInput_1, int outputLength)
{
    tflite::StatefulNnApiDelegate::Options opts;
    opts.accelerator_name = "google-edgetpu";
    opts.accelerator_name = "qti-dsp";

    TfLiteDelegate* nnapiDelegate = new tflite::StatefulNnApiDelegate(opts);

    TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();

    TfLiteInterpreterOptionsAddDelegate(options, nnapiDelegate);

    TfLiteModel* model = TfLiteModelCreateFromFile(model_path);
    TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);

    TfLiteInterpreterAllocateTensors(interpreter);
    auto* inputTensor_0 = TfLiteInterpreterGetInputTensor(interpreter, 0);
    auto* inputTensor_1 = TfLiteInterpreterGetInputTensor(interpreter, 1);

    auto status = TfLiteTensorCopyFromBuffer(inputTensor_0, randomInput_0.data(), randomInput_0.size()*sizeof(float));
    assert(status == kTfLiteOk);

    status = TfLiteTensorCopyFromBuffer(inputTensor_1, randomInput_1.data(), randomInput_1.size()*sizeof(float));
    assert(status == kTfLiteOk);

    TfLiteInterpreterInvoke(interpreter);

    std::vector<float> output(outputLength);
    auto const* outputTensor = TfLiteInterpreterGetOutputTensor(interpreter, 0);

    status = TfLiteTensorCopyToBuffer(outputTensor, output.data(), output.size()*sizeof(float));
    assert(status == kTfLiteOk);

    std::cout<<"\nThe average elapsed time in nnapi delegate: ";
    Benchmark(interpreter);

    TfLiteInterpreterDelete(interpreter);
    delete reinterpret_cast<tflite::StatefulNnApiDelegate*>(nnapiDelegate);
    TfLiteInterpreterOptionsDelete(options);
    TfLiteModelDelete(model);

    return output;
}

int main(int argc, char** argv) {

    cxxopts::Options options("ModelTest", "Test model on mobile device");

    options.add_options()
        ("m,model", "model name (int8 tflite version)", cxxopts::value<std::string>());

    auto result = options.parse(argc, argv);

    if (!result.count("model"))
    {
        throw std::runtime_error("You must provide the model name.");
    }

    auto model = result["model"].as<std::string>();

    if (model == "int8_quantize_concat.tflite")
    {
        std::vector<float> randomInput_0(1*128*128*3);
        std::vector<float> randomInput_1(1*128*128*6);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.f, 1.f); 
        std::generate(randomInput_0.begin(), randomInput_0.end(), [&](){return dis(gen);});
        std::generate(randomInput_1.begin(), randomInput_1.end(), [&](){return dis(gen);});

        nnapi_inference_two_inputs("./model_files/int8_quantize_concat.tflite", randomInput_0, randomInput_1, 1*128*128*64);
    }
    else if (model == "int8_large_Dense.tflite")
    {
        std::vector<float> randomInput(1*4096);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.f, 1.f); 
        std::generate(randomInput.begin(), randomInput.end(), [&](){return dis(gen);});

        nnapi_inference_single_input("./model_files/int8_large_Dense.tflite", randomInput, 1*1024);
    }
    else
    {
        throw std::runtime_error("The model name should be either 'int8_quantize_concat.tflite' or 'int8_large_Dense.tflite'.");
    }
    
}
