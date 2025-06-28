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
            {"path", "i17_1G.zarr"}
        }},
        {"dtype", "float32"},
    };

    auto total_start = std::chrono::high_resolution_clock::now();
    
    auto io_start = std::chrono::high_resolution_clock::now();
    // Open the zarr file
    auto open_future = ts::Open<float, 4>(spec, ts::Context::Default());

    auto result = open_future.result();

    if (!result.ok()) {
        std::cerr << "Failed to open Zarr: " << result.status() << "\n";
        return 1;
    }

    ts::TensorStore<float, 4> store = *result;


    auto read_result = ts::Read(store).result();
    if (!read_result.ok()) {
        std::cerr << "Failed to read: " << read_result.status() << "\n";
        return 1;
    }
    auto array = *read_result;
    // Transform to an 1D array
    float* ptr = array.data();
    std::size_t total_size = array.num_elements();
    std::vector<float> array_1d(ptr, ptr + total_size);

    auto io_end = std::chrono::high_resolution_clock::now();

    std::cout << "Shape: " << array.shape() << "\n";

    //for (tensorstore::Index i = 0; i < std::min(array.shape()[0], tensorstore::Index(5)); ++i) {
    //    for (tensorstore::Index j = 0; j < std::min(array.shape()[1], tensorstore::Index(5)); ++j) {
    //        std::cout << array(i, j) << " ";
    //    }
    //    std::cout << "\n";
    //}
    std::vector<float> spectrum(array.shape()[1], 0.0f);
    std::vector<int> valid_pixels(array.shape()[1], 0);
    using std::isnan;
    auto compute_start = std::chrono::high_resolution_clock::now();
    int channels = array.shape()[1];
    int n_x = array.shape()[2];
    int n_y = array.shape()[3];

    // Calculate the spectrum of each chunk (1, 1, 432, 432)
    long my_pixels = array.shape()[1] * array.shape()[2] * array.shape()[3];
    
    for (long iter = 0; iter < my_pixels; ++iter) {
        if (!isnan(array_1d[iter])) {
            long ch = iter / (n_x * n_y);
            spectrum[ch] += array_1d[iter];
            valid_pixels[ch] += 1;
        
            //if ((iter + 1) % (n_x * n_y) == 0) {
            //    spectrum[ch] /= valid_pixels[ch];
            //}
        }
    }
    for (int ch = 0; ch < channels; ++ch) {
        if (valid_pixels[ch] > 0) {
            spectrum[ch] /= valid_pixels[ch];
    }
}

    // std::transform(std::execution::par,
    //     spectrum.begin(), 
    //     spectrum.end(), 
    //     valid_pixels.begin(),
    //     spectrum.begin(),
    //     [](float spectrum, int count) {
    //         return (count > 0) ? spectrum / count : spectrum;
    //     }
    // );
    auto compute_end = std::chrono::high_resolution_clock::now();
    auto total_end = std::chrono::high_resolution_clock::now();

    // Output the time taken for each step
    double io_time = std::chrono::duration<double>(io_end - io_start).count();
    double compute_time = std::chrono::duration<double>(compute_end - compute_start).count();
    double total_time = std::chrono::duration<double>(total_end - total_start).count();
    outfile << total_time << " " << io_time << " " << compute_time << " sec." << std::endl;

    // output to file
    std::ofstream output("spectrum.txt");
    if (!output.is_open()) {
        std::cerr << "Failed to open output file.\n";
        return 1;
    }

    for (ts::Index ch = 0; ch < array.shape()[1]; ++ch) {
        output << ch << " " << spectrum[ch] << "\n";
    }
    output.close();

    return 0;
}

