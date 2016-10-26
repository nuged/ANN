#pragma once

#include "Helpful Functions.h"

class TNeuron {
public:
    TNeuron() {}
    TNeuron(float input) :
        data(input)
    {}

    void Calculate() {
        ActivativeFunction(data);
    }

    float Out() const {
        return data;
    }

private:
    float data;
};
