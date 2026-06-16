#include <iostream>
#include <memory>

class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t bytes)
        : bytes_(bytes), data_(std::make_unique<unsigned char[]>(bytes)) {
        std::cout << "[alloc] " << bytes_ << " bytes\n";
    }

    ~DeviceBuffer() {
        std::cout << "[free ] " << bytes_ << " bytes\n";
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&&) noexcept = default;
    DeviceBuffer& operator=(DeviceBuffer&&) noexcept = default;

    std::size_t size() const {
        return bytes_;
    }

private:
    std::size_t bytes_ = 0;
    std::unique_ptr<unsigned char[]> data_;
};

int main() {
    DeviceBuffer smem_tile(128 * 256 * sizeof(float));
    std::cout << "buffer size = " << smem_tile.size() << " bytes\n";
    return 0;
}
