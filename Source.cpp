#pragma warning(disable : 4996)

#include <cstdlib>
#include <iostream>
#include "Neural Network.h"
#include "Helpful Functions.h"

int main() {
    TNeuralNetwork nn(784, 300, 1);
    char c;
    std::cout << "Press:\nD - for recognizing the image\nT - for doing test\nL -  for learning:\n";
    std::cin >> c;
    if (c == 'T')
        nn.Test();
    else if (c == 'D')
        nn.Do();
    else if (c == 'L') {
        std::cout << "Enter number of iterations:\n";
        unsigned n;
        std::cin >> n;
        nn.Learn(n);
    }
}
