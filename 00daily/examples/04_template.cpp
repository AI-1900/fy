#include <iostream>
#include <type_traits>

template <typename T>
T add(T a, T b) {
    return a + b;
}

template <typename T, int M, int N>
class StaticMatrix {
public:
    static_assert(M > 0 && N > 0, "matrix shape must be positive");

    constexpr int rows() const { return M; }
    constexpr int cols() const { return N; }
    constexpr int numel() const { return M * N; }

    T data[M * N]{};
};

template <typename T>
void print_type_category() {
    if constexpr (std::is_integral_v<T>) {
        std::cout << "integral type\n";
    } else if constexpr (std::is_floating_point_v<T>) {
        std::cout << "floating point type\n";
    } else {
        std::cout << "other type\n";
    }
}

int main() {
    std::cout << "add<int>(3, 4) = " << add<int>(3, 4) << "\n";
    std::cout << "add<float>(1.5, 2.25) = " << add<float>(1.5f, 2.25f) << "\n";

    StaticMatrix<float, 128, 256> tile;
    std::cout << "tile shape = " << tile.rows() << "x" << tile.cols()
              << ", numel = " << tile.numel() << "\n";

    print_type_category<int>();
    print_type_category<float>();
    return 0;
}
