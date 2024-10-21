cmake_minimum_required(VERSION 3.10)
project(MyProject)


include_directories(/home/pentaery/local/include)
include_directories(eigen-3.4.0)
include_directories(src)
find_package(MKL REQUIRED)
find_package(OpenMP REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(MKL REQUIRED IMPORTED_TARGET mkl-dynamic-lp64-iomp)

add_executable(test src/test.cc src/system.cc)
if(MKL_FOUND)
    target_link_libraries(test ${MKL_LIBRARIES})
    include_directories(${MKL_INCLUDE_DIRS})
    target_compile_definitions(test PRIVATE MKL_ENABLE_OPENMP)
endif()

target_link_libraries(test /home/pentaery/local/lib/libmetis.so)
if(OpenMP_CXX_FOUND)
    target_link_libraries(test OpenMP::OpenMP_CXX)
endif()
