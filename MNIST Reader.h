#pragma once
#include <fstream>
#include<vector>

void Read(std::vector<std::vector<float>>& images, int& magic_number, int& size, int& rows, int& columns,
    std::ifstream& f);

void Read(std::vector<float>& labels, int& magic_number, int& size,
    std::ifstream& f);
