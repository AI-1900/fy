#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

int main() {
    std::vector<int> values{5, 1, 4, 2, 3};

    std::sort(values.begin(), values.end(), [](int a, int b) {
        return a < b;
    });

    int sum = std::accumulate(values.begin(), values.end(), 0);

    std::vector<int> even_values;
    std::copy_if(values.begin(), values.end(), std::back_inserter(even_values), [](int x) {
        return x % 2 == 0;
    });

    std::cout << "sorted: ";
    for (int x : values) {
        std::cout << x << " ";
    }
    std::cout << "\nsum = " << sum << "\n";

    std::cout << "even: ";
    for (int x : even_values) {
        std::cout << x << " ";
    }
    std::cout << "\n";
    return 0;
}
