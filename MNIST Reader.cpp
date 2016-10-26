#include "MNIST Reader.h"
#include "Helpful Functions.h"

void Read(std::vector<std::vector<float>>& images, int& magic_number, int& size, int& rows, int& columns,
    std::ifstream& f) {
    f.read((char *)&magic_number, sizeof(int));
    magic_number = ReverseInt(magic_number);
    f.read((char *)&size, sizeof(int));
    size = ReverseInt(size);
    f.read((char *)&rows, sizeof(int));
    rows = ReverseInt(rows);
    f.read((char *)&columns, sizeof(int));
    columns = ReverseInt(columns);
    images.resize(size);
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < rows*columns; ++j) {
            unsigned char x;
            f.read((char *)&x, sizeof(x));
            images[i].push_back(((float)x) / 255);
        }
    }
}

void Read(std::vector<float>& labels, int& magic_number, int& size,
    std::ifstream& f) {
    f.read((char *)&magic_number, sizeof(int));
    magic_number = ReverseInt(magic_number);
    f.read((char *)&size, sizeof(int));
    size = ReverseInt(size);
    for (size_t i = 0; i < size; ++i) {
        char x;
        f.read((char *)&x, sizeof(x));
        labels.push_back(((float)x) / 9);
    }
}