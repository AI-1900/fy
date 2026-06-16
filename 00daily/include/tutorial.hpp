#pragma once

#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace tutorial {

class Layer {
public:
    virtual ~Layer() = default;
    virtual float forward(float x) const = 0;
    virtual std::string name() const = 0;
};

class Scale final : public Layer {
public:
    explicit Scale(float alpha) : alpha_(alpha) {}

    float forward(float x) const override {
        return alpha_ * x;
    }

    std::string name() const override {
        return "Scale";
    }

private:
    float alpha_ = 1.0f;
};

class Bias final : public Layer {
public:
    explicit Bias(float beta) : beta_(beta) {}

    float forward(float x) const override {
        return x + beta_;
    }

    std::string name() const override {
        return "Bias";
    }

private:
    float beta_ = 0.0f;
};

inline void run_pipeline() {
    std::vector<std::unique_ptr<Layer>> layers;
    layers.emplace_back(std::make_unique<Scale>(2.0f));
    layers.emplace_back(std::make_unique<Bias>(3.0f));

    float x = 4.0f;
    for (const auto& layer : layers) {
        x = layer->forward(x);
        std::cout << layer->name() << " output = " << x << "\n";
    }
}

}  // namespace tutorial
