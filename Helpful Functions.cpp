#include "Helpful Functions.h"
#include <cmath>

float Derivative(float arg) {
        arg = arg * (1 - arg);
        return arg;
}

void ActivativeFunction(float& arg) {
    arg = 1 / (1 + exp(-arg));
}

int ReverseInt(int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}
