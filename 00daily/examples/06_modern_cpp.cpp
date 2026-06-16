#include <iostream>
#include <optional>
#include <string_view>
#include <tuple>

constexpr int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

std::optional<int> parse_positive_int(int x) {
    if (x > 0) {
        return x;
    }
    return std::nullopt;
}

void print_kernel_name(std::string_view name) {
    std::cout << "kernel = " << name << "\n";
}

int main() {
    constexpr int block_m = 128;
    constexpr int problem_m = 4096;
    constexpr int grid_m = ceil_div(problem_m, block_m);
    static_assert(grid_m == 32);

    auto [m, n, k] = std::tuple<int, int, int>{128, 256, 64};
    std::cout << "shape = (" << m << ", " << n << ", " << k << ")\n";

    print_kernel_name("bf16_gemm");

    auto maybe_value = parse_positive_int(42);
    if (maybe_value.has_value()) {
        std::cout << "optional value = " << *maybe_value << "\n";
    }

    std::cout << "grid_m = " << grid_m << "\n";
    return 0;
}
