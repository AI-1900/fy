# C++ 核心特性学习笔记

## 一、回答目标

本仓库目标是用最小 Linux CMake 工程展示 C++ 核心特性：

- 类与对象；
- 构造函数与析构函数；
- RAII；
- 智能指针；
- 继承与虚函数；
- 多态；
- 模板；
- STL；
- lambda；
- constexpr；
- optional / string_view；
- move semantics。

## 二、核心结论

C++ 核心不是语法堆叠，而是三条主线：

| 主线 | 关键特性 | 工程价值 |
|---|---|---|
| 对象生命周期 | 构造、析构、RAII | 资源安全，不泄漏 |
| 抽象与复用 | 继承、虚函数、模板 | 统一接口，减少重复代码 |
| 性能与表达力 | STL、lambda、constexpr、move | 写法简洁，运行高效 |

## 三、核心特性关系图

```text
C++ Core
├── class/object
│   ├── constructor
│   ├── destructor
│   └── member function
├── RAII
│   ├── unique_ptr
│   └── shared_ptr
├── polymorphism
│   ├── base class
│   ├── virtual function
│   └── vtable dispatch
├── template
│   ├── function template
│   ├── class template
│   └── if constexpr
├── STL
│   ├── vector
│   ├── algorithm
│   └── iterator
└── modern C++
    ├── constexpr
    ├── optional
    ├── string_view
    └── move semantics
```

## 四、学习建议

1. 先运行所有示例；
2. 修改参数，观察输出变化；
3. 用 `gdb` 单步查看对象生命周期；
4. 用 `nm -C` 看虚函数符号；
5. 用 `objdump -d -C` 看函数调用和虚调用差异。
