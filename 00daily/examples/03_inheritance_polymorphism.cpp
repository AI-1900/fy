#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

class Op {
public:
    virtual ~Op() = default;
    virtual float compute(float x) const = 0;
    virtual const char* name() const = 0;
};

class Relu : public Op {
public:
    float compute(float x) const override {
        return x > 0.0f ? x : 0.0f;
    }
    const char* name() const override {
        return "Relu";
    }
};

class Sigmoid : public Op {
public:
    float compute(float x) const override {
        return 1.0f / (1.0f + std::exp(-x));
    }
    const char* name() const override {
        return "Sigmoid";
    }
};

int main() {
    std::vector<std::unique_ptr<Op>> ops;
    ops.emplace_back(std::make_unique<Relu>());
    ops.emplace_back(std::make_unique<Sigmoid>());

    for (const auto& op : ops) {
        std::cout << op->name() << "(-1.5) = " << op->compute(-1.5f) << "\n";
    }
    return 0;
}
