#include <iostream>
#include <string>
#include <vector>

class KernelConfig {
public:
    explicit KernelConfig(std::string name)
        : name_(std::move(name)) {
        std::cout << "[ctor] " << name_ << "\n";
    }

    KernelConfig(const KernelConfig& other)
        : name_(other.name_) {
        std::cout << "[copy] " << name_ << "\n";
    }

    KernelConfig(KernelConfig&& other) noexcept
        : name_(std::move(other.name_)) {
        std::cout << "[move] " << name_ << "\n";
    }

    const std::string& name() const {
        return name_;
    }

private:
    std::string name_;
};

int main() {
    std::vector<KernelConfig> configs;
    configs.reserve(2);

    KernelConfig cfg("sm100_bf16_gemm");
    configs.push_back(cfg);             // copy
    configs.push_back(std::move(cfg));  // move

    for (const auto& item : configs) {
        std::cout << "config = " << item.name() << "\n";
    }
    return 0;
}
