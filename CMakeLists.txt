cmake_minimum_required(VERSION 3.30)
project(image_splitter CXX)

add_executable(image_splitter main.cc)

set(INCLUDEDIRS ${PROJECT_SOURCE_DIR})

if (WIN32)
    find_package(Stb REQUIRED)
    set(INCLUDEDIRS ${INCLUDEDIRS} ${Stb_INCLUDE_DIR})
    if (DEFINED ORT_ROOT_DIR)
        message(STATUS "OnnxRuntime enabled.")
        set(INCLUDEDIRS ${INCLUDEDIRS} ${ORT_ROOT_DIR}/include)
        set(DEPENDENCIES ${DEPENDENCIES} ${ORT_ROOT_DIR}/lib/onnxruntime.lib)
    else()
        message(STATUS "OnnxRuntime disabled.")
    endif()
else()
    find_package(PkgConfig)
    pkg_check_modules(STB QUIET stb)
    set(INCLUDEDIRS ${INCLUDEDIRS} ${STB_INCLUDE_DIRS})
    set(DEPENDENCIES ${DEPENDENCIES} ${STB_LIBRARIES})
    find_package(onnxruntime REQUIRED)
    set(DEPENDENCIES ${DEPENDENCIES} onnxruntime::onnxruntime)
endif()

target_compile_options(image_splitter PUBLIC -O2 -Wall)
target_compile_features(image_splitter PUBLIC cxx_std_20)

target_include_directories(image_splitter PUBLIC ${INCLUDEDIRS})

target_link_libraries(image_splitter PRIVATE ${DEPENDENCIES})
