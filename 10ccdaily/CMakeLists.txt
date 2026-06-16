cmake_minimum_required(VERSION 3.20)

project(cpp_core_features_repo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

add_compile_options(
    -Wall
    -Wextra
    -Wpedantic
    -Werror
)

add_executable(01_class_object examples/01_class_object.cpp)
add_executable(02_raii_unique_ptr examples/02_raii_unique_ptr.cpp)
add_executable(03_inheritance_polymorphism examples/03_inheritance_polymorphism.cpp)
add_executable(04_template examples/04_template.cpp)
add_executable(05_stl_lambda examples/05_stl_lambda.cpp)
add_executable(06_modern_cpp examples/06_modern_cpp.cpp)
add_executable(07_move_semantics examples/07_move_semantics.cpp)
add_executable(08_all_in_one src/main.cpp)

target_include_directories(08_all_in_one PRIVATE include)
