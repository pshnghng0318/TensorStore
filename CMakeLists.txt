cmake_minimum_required(VERSION 3.27)
project(read_zarr_example LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

FetchContent_Declare(
  tensorstore
  URL "https://github.com/google/tensorstore/archive/refs/tags/v0.1.75.tar.gz"
)

FetchContent_MakeAvailable(tensorstore)

add_executable(read_zarr read_zarr.cpp)

target_link_libraries(
  read_zarr
  PRIVATE
    tensorstore::tensorstore tensorstore::all_drivers
)
