#include "Neural Network.h"
#include "Helpful Functions.h"
#include "MNIST Reader.h"
#include <iostream>
#include <iomanip>
#include <random>

void TNeuralNetwork::InitializeWeights() {
    float scale_factor;
    size_t in_size = input.size();
    size_t hid_size = hidden.size();
    size_t out_size = output.size();
    scale_factor = 0.7 * pow(hid_size, 1 / in_size);

    // randomly fill Input-Hidden connections with values between -0.5 and 0.5
    std::random_device rd;
    std::mt19937_64 generator(rd());
    std::uniform_real_distribution<float> distribution(-0.5, 0.5);
    for (size_t i = 0; i < hid_size; ++i) {
        for (size_t j = 0; j < in_size; ++j) {
            float weight = distribution(generator);
            Inp_Hid_Weights[i].push_back(weight);
        }
    }

    // correcting input weights
    std::vector<float> lengthes;
    for (size_t i = 0; i < hid_size; ++i) {
        float length = 0;
        for (size_t j = 0; j < in_size; ++j)
            length += pow(Inp_Hid_Weights[i][j], 2);
        length = sqrt(length);
        lengthes.push_back(length);
    }
    for (size_t i = 0; i < hid_size; ++i) {
        for (size_t j = 0; j < in_size; ++j) {
            Inp_Hid_Weights[i][j] = scale_factor * Inp_Hid_Weights[i][j] / lengthes[i];
        }
    }

    // randomly initialize input bias with values between -scale_factor and scale_factor
    std::uniform_real_distribution<float> dist(-scale_factor, +scale_factor);
    for (size_t i = 0; i < hid_size; ++i) {
        float weight = dist(generator);
        inp_bias.push_back(weight);
    }

    // randomly initialize Hidden-Output connections with values between -0.5 and 0.5
    for (size_t i = 0; i < out_size; ++i) {
        for (size_t j = 0; j < hid_size; ++j) {
            float weight = distribution(generator);
            Hid_Out_Weights[i].push_back(weight);
        }
    }

    // randomly fill hidden bias with values between -0.5 and 0.5
    for (size_t i = 0; i < out_size; ++i) {
        float weight = distribution(generator);
        hid_bias.push_back(weight);
    }
}

void TNeuralNetwork::EnterData() {
    std::vector<std::vector<float>> images;
    int magic = 0, size = 0, rows = 0, columns = 0;
    std::ifstream f;
    f.open("Image", std::ios::binary);
    Read(images, magic, size, rows, columns, f);
    f.close();
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = images[0][i];
    }
}

void TNeuralNetwork::SaveToFile() {
    std::ofstream fout;
    fout.open("NN.dat");
    for (auto elem : Inp_Hid_Weights) {
        for (auto el : elem)
            fout << el << " ";
        fout << "\n";
    }
    fout << "\n";
    for (auto elem : inp_bias)
        fout << elem << " ";
    fout << "\n\n";
    for (auto elem : Hid_Out_Weights) {
        for (auto el : elem)
            fout << el << " ";
        fout << "\n";
    }
    fout << "\n";
    for (auto elem : hid_bias)
        fout << elem << " ";
    fout << "\n";
    fout.close();
}

void TNeuralNetwork::ReadFromFile() {
    std::ifstream fin;
    fin.open("NN.dat");
    for (size_t i = 0; i < hidden.size(); ++i) {
        for (size_t j = 0; j < input.size(); ++j) {
            float x;
            fin >> x;
            Inp_Hid_Weights[i].push_back(x);
        }
    }
    for (size_t i = 0; i < hidden.size(); ++i) {
        float x;
        fin >> x;
        inp_bias.push_back(x);
    }
    for (size_t i = 0; i < output.size(); ++i) {
        for (size_t j = 0; j < hidden.size(); ++j) {
            float x;
            fin >> x;
            Hid_Out_Weights[i].push_back(x);
        }
    }
    for (size_t i = 0; i < output.size(); ++i) {
        float x;
        fin >> x;
        hid_bias.push_back(x);
    }
    fin.close();
}

void TNeuralNetwork::Process() {
    //Transmition data to hidden layer and calculation
    for (size_t i = 0; i < hidden.size(); ++i) {
        float sum = 0;
        for (size_t j = 0; j < input.size(); ++j) {
            sum += input[j].Out() * Inp_Hid_Weights[i][j];
        }
        sum += inp_bias[i];
        hidden[i] = sum;
        hidden[i].Calculate();
    }

    //Transmition data to output layer and calculation
    for (size_t i = 0; i < output.size(); ++i) {
        float sum = 0;
        for (size_t j = 0; j < hidden.size(); ++j) {
            sum += hidden[j].Out() * Hid_Out_Weights[i][j];
        }
        sum += hid_bias[i];
        output[i] = sum;
        output[i].Calculate();
    }
}

void TNeuralNetwork::Learn(size_t iterations) {
    //Randomly initialize weights
    InitializeWeights();

    size_t inp_size = input.size();
    size_t hid_size = hidden.size();
    size_t out_size = output.size();

    //Back propagation:

    //Copy learning data
    std::cerr << "copying\n";
    std::vector<std::vector<float>> examples;
    std::vector<float> targets;
    int DataSize = 0, rows = 0, columns = 0, magic = 0;
    {std::ifstream f;
    f.open("data/Train images", std::ios::binary);
    Read(examples, magic, DataSize, rows, columns, f);
    f.close();
    f.open("data/Train labels", std::ios::binary);
    Read(targets, magic, DataSize, f);
    f.close();
    std::cerr << "copying finished\n"; }

    float coef = 0.5;
    //learning
    for (size_t iter = 0; iter < iterations; ++iter) {
        float MSE = 0;
        // std::ofstream fout("result.txt");
        for (size_t it = 0; it < DataSize; ++it) {
            //inputting of learning data
            
            for (size_t i = 0; i < input.size(); ++i) {
                input[i] = examples[it][i];
            }
            // processing NN
            Process();
            // fout << "Answer is: " << round(targets[it] * 9) << "\n";
            // fout << "My Answer is: " << round(output[0].Out() * 9) << "\n";

            TBias delta_hid_bias;
            TBias delta_inp_bias;
            std::vector<float> delta_out(out_size);
            std::vector<float> delta_hid(hid_size);
            TWeights delta_hid_out(out_size);
            TWeights delta_inp_hid(hid_size);

            MSE += pow(9 * (targets[it] - output[0].Out()), 2) / (float) DataSize;

            // calcuating output errors
            for (size_t i = 0; i < out_size; ++i) {
                delta_out[i] = Derivative(output[i].Out()) * ((float)targets[it] - output[i].Out());
                // delta_out[i] = ((float)targets[it] - output[i].Out());
            }

            // calculating values for correction Hid-Out weights
            for (size_t i = 0; i < out_size; ++i) {
                for (size_t j = 0; j < hid_size; ++j) {
                    float delta = coef * delta_out[i] * hidden[j].Out();
                    delta_hid_out[i].push_back(delta);
                }
            }

            // calculating values for correction hidden bias
            for (size_t i = 0; i < out_size; ++i) {
                float delta = coef * delta_out[i];
                delta_hid_bias.push_back(delta);
            }

            // calculatig hidden layer errors
            for (size_t i = 0; i < hid_size; ++i) {
                float sum = 0;
                for (size_t j = 0; j < out_size; ++j) {
                    sum += delta_out[j] * Hid_Out_Weights[j][i];
                }
                delta_hid[i] = Derivative(hidden[i].Out()) * sum;
            }

            // calculating values for correction Inp_Hid weights
            for (size_t i = 0; i < hid_size; ++i) {
                for (size_t j = 0; j < inp_size; ++j) {
                    float delta = coef * delta_hid[i] * input[j].Out();
                    delta_inp_hid[i].push_back(delta);
                }
            }

            // calculating values for correction input bias
            for (size_t i = 0; i < hid_size; ++i) {
                float delta = coef * delta_hid[i];
                delta_inp_bias.push_back(delta);
            }

            // correction of all the weights and biases
            for (size_t i = 0; i < out_size; ++i) {
                for (size_t j = 0; j < hid_size; ++j) {
                    Hid_Out_Weights[i][j] += delta_hid_out[i][j];
                }
                hid_bias[i] += delta_hid_bias[i];
            }
            for (size_t i = 0; i < hid_size; ++i) {
                for (size_t j = 0; j < inp_size; ++j)
                    Inp_Hid_Weights[i][j] += delta_inp_hid[i][j];
                inp_bias[i] += delta_inp_bias[i];
            }
        }
        std::cout << "iterations: " << iter << "\tMSE: " << std::fixed << std::setprecision(6) << MSE << "\n";
        // fout.close();
    }
    SaveToFile();
}

void TNeuralNetwork::Do() {
    size_t inp_size = input.size();
    size_t hid_size = hidden.size();
    size_t out_size = output.size();
    ReadFromFile();
    // Enter data for input layer
    EnterData();
    Process();
    PrintData();
}

void TNeuralNetwork::PrintData() {
    for (auto elem : output) {
        std::cout << round(9*elem.Out()) << "\n";
    }
}

void TNeuralNetwork::Test() {
    ReadFromFile();
    std::vector<std::vector<float>> test_images;
    std::vector<float> test_labels;
    int magic = 0, size = 0, rows = 0, columns = 0;
    {std::ifstream f;
    f.open("data/Test images", std::ios::binary);
    Read(test_images, magic, size, rows, columns, f);
    f.close();
    f.open("data/Test labels", std::ios::binary);
    Read(test_labels, magic, size, f);
    f.close(); }
    std::ofstream fout("result.txt");
    float sum = 0;
    unsigned s = 0;
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < rows; ++j) {
            for (size_t k = 0; k < columns; ++k) {
                fout << (char)(255*test_images[i][j*columns + k]) << " ";
            }
            fout << "\n";
        }
        fout << "Answer is: " << round(test_labels[i] * 9) << "\n";
        for (size_t j = 0; j < input.size(); ++j) {
            input[j] = test_images[i][j];
        }
        Process();
        fout << "My Answer is: " << round(output[0].Out() * 9) << "\n";
        if (round(test_labels[i] * 9) != round(output[0].Out() * 9))
            ++s;
        sum += pow(9 * (output[0].Out() - test_labels[i]), 2) / size;
    }
    fout << std::fixed << std::setprecision(6) << "Error is " << sum << "\n";
    fout << "errors " << s;
    fout.close();
}
