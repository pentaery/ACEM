find_package(OpenMP REQUIRED)


add_executable(test HelloOpenMP.c)

if(OpenMP_CXX_FOUND)
    target_link_libraries(test OpenMP::OpenMP_CXX)
endif()