# C++ Core Features Minimal Linux Repo

本仓库用于从零开始学习 C++ 核心特性，所有示例均为最小可运行代码，适合在 Linux 端直接编译运行。

## 1. 覆盖内容

| 示例 | 可执行文件 | 主题 | 工程意义 |
|---|---|---|---|
| 01 | `01_class_object` | class / 构造 / 析构 / 成员函数 | 理解对象生命周期 |
| 02 | `02_raii_unique_ptr` | RAII / `std::unique_ptr` | 自动资源管理，替代手动 `new/delete` |
| 03 | `03_inheritance_polymorphism` | 继承 / 虚函数 / 多态 | 基类接口 + 派生类实现 |
| 04 | `04_template` | 函数模板 / 类模板 / `if constexpr` | 编译期泛型编程 |
| 05 | `05_stl_lambda` | STL / lambda / algorithm | 标准库容器和算法 |
| 06 | `06_modern_cpp` | `constexpr` / `optional` / `string_view` / structured binding | 现代 C++ 常用写法 |
| 07 | `07_move_semantics` | 拷贝 / 移动语义 / `std::move` | 避免不必要拷贝 |
| 08 | `08_all_in_one` | 综合例子 | 类 + RAII + 多态 + STL |

## 2. 环境要求

```bash
sudo apt update
sudo apt install -y build-essential cmake
```

要求：

- GCC >= 9
- CMake >= 3.20
- Linux x86_64 / aarch64 均可

## 3. 编译

```bash
./scripts/build.sh
```

等价命令：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j$(nproc)
```

## 4. 运行所有示例

```bash
./scripts/run_all.sh
```

单独运行：

```bash
./build/01_class_object
./build/02_raii_unique_ptr
./build/03_inheritance_polymorphism
./build/04_template
./build/05_stl_lambda
./build/06_modern_cpp
./build/07_move_semantics
./build/08_all_in_one
```

## 5. 学习路径

```text
class / struct
  ↓
constructor / destructor
  ↓
RAII / unique_ptr
  ↓
inheritance / virtual / polymorphism
  ↓
template / constexpr
  ↓
STL / lambda / algorithm
  ↓
move semantics
  ↓
CMake project
  ↓
CUDA C++ / CUTLASS / DeepGEMM
```

## 6. 核心执行模型

```text
main.cpp
  |
  v
std::vector<std::unique_ptr<Layer>>
  |
  +--> Scale::forward
  |
  +--> Bias::forward
  |
  v
output
```

## 7. 重点调试命令

```bash
# 查看符号
nm -C build/03_inheritance_polymorphism | grep Op

# 反汇编
objdump -d -C build/03_inheritance_polymorphism | less

# gdb 调试
gdb ./build/03_inheritance_polymorphism
```

GDB 中可尝试：

```gdb
break main
run
next
print ops.size()
info vtbl *ops[0].get()
```
