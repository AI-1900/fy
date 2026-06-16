#include <iostream>
#include <string>

class Tensor {
public:
    Tensor(std::string name, int rows, int cols)
        : name_(std::move(name)), rows_(rows), cols_(cols) {
        std::cout << "[ctor] Tensor " << name_ << " constructed\n";
    }

    ~Tensor() {
        std::cout << "[dtor] Tensor " << name_ << " destructed\n";
    }

    int numel() const {
        return rows_ * cols_;
    }

    void print() const {
        std::cout << "Tensor{name=" << name_
                  << ", shape=(" << rows_ << ", " << cols_ << ")"
                  << ", numel=" << numel() << "}\n";
    }

private:
    std::string name_;
    int rows_ = 0;
    int cols_ = 0;
};

int main() {
    Tensor a("A", 128, 256);
    a.print();
    return 0;
}
