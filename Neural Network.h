#pragma once
#include "Neuron.h"
#include <vector>
#include <fstream>

typedef std::vector<TNeuron> TNeuralLayer;
typedef std::vector<std::vector<float>> TWeights;
typedef std::vector<float> TBias;

class TNeuralNetwork {
public:
    TNeuralNetwork(unsigned int inp, unsigned int hid, unsigned int out) :
        input(inp),
        hidden(hid),
        output(out),
        Inp_Hid_Weights(hid),
        Hid_Out_Weights(out)
    {}
    void Do();
    void Learn(size_t iterations);
    void Test();

private:
    TNeuralLayer input;
    TNeuralLayer hidden;
    TNeuralLayer output;
    TWeights Inp_Hid_Weights;
    TWeights Hid_Out_Weights;
    TBias inp_bias;
    TBias hid_bias;
    std::ifstream Learning_in;
    void EnterData();
    void InitializeWeights();
    void SaveToFile();
    void ReadFromFile();
    void PrintData();
    void Process();
};
