cmake_minimum_required(VERSION 3.27)
#project(read_zarr_example LANGUAGES CXX)
project(histogram_example LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable MPI
find_package(MPI REQUIRED)

# Fetch TensorStore from GitHub
include(FetchContent)

FetchContent_Declare(
  tensorstore
  URL "https://github.com/google/tensorstore/archive/refs/tags/v0.1.75.tar.gz"
)

FetchContent_MakeAvailable(tensorstore)

# Executable
#add_executable(read_zarr read_zarr.cpp)
add_executable(histogram histogram.cpp)

# Link against TensorStore + drivers + MPI
#target_link_libraries(read_zarr
target_link_libraries(histogram
  PRIVATE
    tensorstore::tensorstore
    tensorstore::all_drivers
    MPI::MPI_CXX
)

# Optional: include MPI headers explicitly
#target_include_directories(read_zarr PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_include_directories(histogram PRIVATE ${MPI_CXX_INCLUDE_DIRS})


add_executable(spectrum spectrum.cpp)
target_link_libraries(spectrum
  PRIVATE
    tensorstore::tensorstore
    tensorstore::all_drivers
    MPI::MPI_CXX
)
target_include_directories(spectrum PRIVATE ${MPI_CXX_INCLUDE_DIRS})
