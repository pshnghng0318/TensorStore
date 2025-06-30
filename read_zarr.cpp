#include <tensorstore/tensorstore.h>
#include <tensorstore/open.h>
#include <tensorstore/util/result.h>
#include <tensorstore/context.h>
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
#include <chrono>
#include <execution>
#include <vector>
#include <cmath>
#include <algorithm>    

namespace ts = tensorstore;

int main() {
    std::ofstream outfile("result1.dat", std::ios::app);
    nlohmann::json spec = {
        {"driver", "zarr"},
        {"kvstore", {
            {"driver", "file"},
            {"path", "alma16G.zarr"}
        }},
        {"dtype", "float32"},
    };
    
    // Open
    auto open_future = ts::Open<float, 4>(spec, ts::Context::Default());
    auto result = open_future.result();
    if (!result.ok()) {
        std::cerr << "Failed to open Zarr: " << result.status() << "\n";
        return 1;
    }
    ts::TensorStore<float, 4> store = *result;

    // Shapes & number of elements
    auto shape = store.domain().shape();
    //auto shape = std::array<ts::Index, 4>{1, 59516, 256, 256};
    std::array<ts::Index, 4> chunk_shape = {1, 1024, 256, 256}; 
    int freq_step = chunk_shape[1];
    std::size_t total_elements = 1;
    for (auto dim : chunk_shape) {
        total_elements *= dim;
    }
    std::cout << "Total elements in a chunk: " << total_elements << "\n";

    // Create an 1D array with lens of a chunk
    std::vector<float> array_1d(total_elements);
    
    // Initializing specs for chunks
    nlohmann::json spec_chunk[shape[1]/freq_step + 1];
    std::vector<float> spectrum(shape[1], 0.0f);
    std::vector<int> valid_pixels(shape[1], 0);
    int channels = shape[1];
    int n_x = shape[2];
    int n_y = shape[3];

    // Initializing timer
    double io_time = 0;
    double compute_time = 0;
    double total_time = 0;
    // If isnan is not defined
    using std::isnan;

    for (int i_start = 0; i_start < shape[1]/freq_step; ++i_start) {
        // Calculate the frequency range for each chunk
        int f_start = i_start * freq_step;
        int f_end = std::min<int>(f_start + freq_step, shape[1]);
        int f_len = f_end - f_start;

        // setup spec for each chunk
        spec_chunk[i_start] = {
            {"driver", "zarr"},
            {"kvstore", {
                {"driver", "file"},
                {"path", "alma16G.zarr"}
            }},
            {"dtype", "float32"},
            {"transform", {
                {"input_shape", {1, chunk_shape[1], shape[2], shape[3]}},
                {"output", {
                    {
                        {"input_dimension", 0},
                        {"offset", 0},
                        {"stride", 1}
                    },
                    {
                        {"input_dimension", 1},
                        {"offset", f_start},
                        {"stride", 1}
                    },
                    {
                        {"input_dimension", 2},
                        {"offset", 0},
                        {"stride", 1}
                    },
                    {
                        {"input_dimension", 3},
                        {"offset", 0},
                        {"stride", 1}
                    }
                }}
            }}
        };

        // Reading zarr
        auto total_start = std::chrono::high_resolution_clock::now();
        auto io_start = std::chrono::high_resolution_clock::now();
        auto open_future_chunk = ts::Open<float, 4>(spec_chunk[i_start], ts::Context::Default());
        auto result_chunk = open_future_chunk.result();
        if (!result_chunk.ok()) {
            std::cerr << "Failed to open Zarr: " << result_chunk.status() << "\n";
            return 1;
        }
        ts::TensorStore<float, 4> store_chunk = *result_chunk;
        auto slice_result = ts::Read(store_chunk).result();
        if (!slice_result.ok()) {
            std::cerr << "Read failed: " << slice_result.status() << "\n";
            continue;
        }
        auto array_chunk = *slice_result;

        // Catch a 1D array by pointers
        float* ptr_chunk = array_chunk.data();
        std::size_t total_size_chunk = array_chunk.num_elements();
        //array_1d.insert(array_1d.end(), ptr_chunk, ptr_chunk + total_size_chunk);
        array_1d = std::vector<float>(ptr_chunk, ptr_chunk + total_size_chunk);

        auto io_end = std::chrono::high_resolution_clock::now();
        io_time += std::chrono::duration<double>(io_end - io_start).count();

        // Calculate specturm
        auto compute_start = std::chrono::high_resolution_clock::now();
        long my_pixels = chunk_shape[1] * shape[2] * shape[3];
        for (long iter = 0; iter < my_pixels; ++iter) {
            if (!isnan(array_1d[iter])) {
                long ch = (iter+i_start*my_pixels) / (n_x * n_y);
                spectrum[ch] += array_1d[iter];
                valid_pixels[ch] += 1;
                if ((iter + 1) % (n_x * n_y) == 0) {
                    spectrum[ch] /= valid_pixels[ch];
                }
            }
        }
        // clear array
        std::vector<float>().swap(array_1d);

        auto compute_end = std::chrono::high_resolution_clock::now();
        compute_time += std::chrono::duration<double>(compute_end - compute_start).count();
        auto total_end = std::chrono::high_resolution_clock::now();
        total_time += std::chrono::duration<double>(total_end - total_start).count();
        std::cout << total_time << " " << io_time << " " << compute_time << " sec." << std::endl;
    }

    // Output the time taken for each step
    outfile << total_time << " " << io_time << " " << compute_time << " sec." << std::endl;

    // Output spectrum
    std::ofstream output("spectrum.dat");
    if (!output.is_open()) {
        std::cerr << "Failed to open output file.\n";
        return 1;
    }
    for (ts::Index ch = 0; ch < shape[1]; ++ch) {
        output << ch << " " << spectrum[ch] << "\n";
    }
    output.close();

    return 0;
}

