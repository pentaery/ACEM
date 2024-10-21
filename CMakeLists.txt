cmake_minimum_required(VERSION 3.10)
project(MyProject)


include_directories(/usr/local/Cellar/metis/5.1.0/include)
find_package(MKL REQUIRED)


add_executable(test src/test.cc)
if(MKL_FOUND)
    target_link_libraries(test ${MKL_LIBRARIES})
    include_directories(${MKL_INCLUDE_DIRS})
endif()

target_link_libraries(test /usr/local/Cellar/metis/5.1.0/lib/libmetis.dylib)
