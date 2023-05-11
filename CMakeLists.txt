cmake_minimum_required(VERSION 3.23)
project(metodos_tp2)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

set(CMAKE_CXX_STANDARD 17)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb -std=c++17 -Wall -Wpedantic -Wextra -Wshadow -Wpointer-arith -Wcast-qual")
set(SOURCE_FILES src/cpp/Matrix.cpp src/cpp/Eigenpair.cpp src/cpp/IO.cpp)
add_executable(metodos_tp2 src/cpp/main.cpp src/cpp/Matrix.cpp src/cpp/Matrix.h src/cpp/Eigenpair.cpp src/cpp/Eigenpair.h src/cpp/IO.cpp src/cpp/IO.h)
target_link_libraries (metodos_tp2 Eigen3::Eigen)
