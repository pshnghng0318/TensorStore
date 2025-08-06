#include <tensorstore/tensorstore.h>
#include <tensorstore/open.h>
#include <tensorstore/util/result.h>
#include <tensorstore/cast.h>
#include <tensorstore/context.h>
#include <tensorstore/array.h>
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <cmath>
#include <omp.h>

namespace ts = tensorstore;

int main(int argc, char **argv) {
    int chunk_size[4] = {1, 64, 256, 256};
    // int chunk_size[4] = {1, 1, 256, 256};

    const std::string zarr_path = "as_128_3D_lz4.zarr/data";
    // const std::string zarr_path = "as_128_3D.zarr/data";
    // const std::string zarr_path = "as_128_3D_lz4_1chChunk.zarr/data";
    // const std::string zarr_path = "as_128_3D_1chChunk.zarr/data";

    int file_num[4] = {0, 0, 0, 0};
    std::string file_num_string;
    bool compressor = false;

    constexpr int zarr_dim = 4;
    std::ofstream outfile("spectrum_result1_12G_nonmpi.dat", std::ios::app);

    const int bitpix = 4; // float32 = 4 bytes

    ts::Index npixels = 0, n_channels = 0, n_x = 0, n_y = 0;
    ts::Index less_pixels = 0;

    std::vector<ts::Index> shape_vec(zarr_dim, 0);

    double io_time = 0.0;
    double compute_time = 0.0;
    double total_time = 0.0;

    auto total_start = std::chrono::high_resolution_clock::now();
    auto io_start = std::chrono::high_resolution_clock::now();

    std::vector<ts::Index> shape(zarr_dim, 0);

    nlohmann::json spec = {
        {"driver", "zarr"},
        {"kvstore", {{"driver", "file"}, {"path", zarr_path}}},
        {"dtype", "float32"},
    };

    nlohmann::json context_spec = {
        {"cache_pool", {{"total_bytes_limit", 64 << 20}}}};
    auto context_result = ts::Context::FromJson(context_spec);
    if (!context_result.ok()) {
        std::cerr << "Failed to create context: " << context_result.status() << "\n";
        return 1;
    }
    auto context = *context_result;
    auto result_shape = ts::Open<float, zarr_dim>(spec, context).result();
    if (!result_shape.ok()) {
        std::cerr << "Open failed: " << result_shape.status() << "\n";
    }

    auto store = *result_shape;
    auto shape_span = store.domain().shape();
    shape.assign(shape_span.begin(), shape_span.end());

    n_channels = shape[1];
    n_x = shape[zarr_dim - 2];
    n_y = shape[zarr_dim - 1];
    std::cout << "Channels: " << n_channels << ", X: " << n_x << ", Y: " << n_y << "\n";
    npixels = ts::Index(n_channels) * n_x * n_y;

    if (zarr_dim == 5) {
        shape_vec = {shape[0], chunk_size[1], shape[2], chunk_size[3], chunk_size[4]};
    } else if (zarr_dim == 4) {
        shape_vec = {shape[0], chunk_size[1], chunk_size[2], chunk_size[3]};
    } else {
        std::cerr << "Unsupported Zarr dimension: " << zarr_dim << "\n";
    }

    auto spec_result = store.spec(); // spec_result 是 Result<Spec>
    if (!spec_result.ok()) {
        std::cerr << "Failed to get spec: " << spec_result.status() << "\n";
    } else {
        auto spec_json_result = spec_result->ToJson();
        if (!spec_json_result.ok()) {
            std::cerr << "Failed to convert spec to JSON: " << spec_json_result.status() << "\n";
        } else {
            const auto &spec_json = *spec_json_result;
            // std::cout << spec_json << std::endl;

            if (spec_json.contains("metadata")) {
                const auto &metadata = spec_json["metadata"];
                if (metadata.contains("compressor") && !metadata["compressor"].is_null()) {
                    // std::cout << "Compressor: " << metadata["compressor"]["id"] << "\n";
                    compressor = true;
                } else {
                    // std::cout << "No compressor (raw, uncompressed chunks)\n";
                    compressor = false;
                }
            } else {
                std::cout << "No metadata found in spec\n";
            }
        }
    }
    store = {};

    // auto l_x = shape[3] / shape_vec[3] + ((shape[3] % shape_vec[3]) ? 1 : 0); // 30 + 1
    // auto l_y = shape[4] / shape_vec[4] + ((shape[4] % shape_vec[4]) ? 1 : 0); // 18 + 1
    auto l_x = shape[2] / shape_vec[2] + ((shape[2] % shape_vec[2]) ? 1 : 0); // 30 + 1
    auto l_y = shape[3] / shape_vec[3] + ((shape[3] % shape_vec[3]) ? 1 : 0); // 18 + 1
    // 64 channels (spectrum)
    auto l_z = shape[1] / shape_vec[1] + ((shape[1] % shape_vec[1]) ? 1 : 0); // 2
    // auto l_z = 1;

    // 64 channels (spectrum)
    ///// 
    ts::Index total_chunks = l_x * l_y * l_z; // 31 * 19 * 2 = 1178
    ts::Index total_loops = total_chunks;

    int xy_chunks = l_x * l_y; // 31 * 19 = 589

    bool no_chunk = false;
    ts::Index my_channels = 1;

    // MPI_Barrier(MPI_COMM_WORLD);
    auto io_end = std::chrono::high_resolution_clock::now();
    io_time += std::chrono::duration<double>(io_end - io_start).count();

    std::vector<double> local_spectrum(n_channels, 0.0);
    std::vector<long> valid_pixels(n_channels, 0);

    #pragma omp parallel
    {
        std::vector<double> thread_spectrum(n_channels, 0.0);
        std::vector<long> thread_valid(n_channels, 0);

        #pragma omp for schedule(dynamic)
        for (ts::Index nloop = 0; nloop < total_loops; ++nloop) {

            std::cout << "Processing loop " << nloop + 1 << "th of " << total_loops << "\n";

            // Time I/O
            auto io_start = std::chrono::high_resolution_clock::now();

            int ith_chunk = nloop;
            // 64 channels (spectrum)
            my_channels = 64;

            if (ith_chunk >= total_chunks) {
                no_chunk = true;
                my_channels = 0;
            } else {
                no_chunk = false;
            }
            ts::Index ch_start = (ith_chunk / xy_chunks) * shape_vec[1];
            // std::cout << "ith_chunk: " << ith_chunk << ", ch_start: " << ch_start << ", my_channels: " << my_channels << std::endl;

            if (!no_chunk) {
                // ts::Index px_start = ((ith_chunk % xy_chunks) % l_x) * shape_vec[3];
                // ts::Index py_start = ((ith_chunk % xy_chunks) / l_x) * shape_vec[4];
                ts::Index px_start = ((ith_chunk % xy_chunks) % l_x) * shape_vec[2];
                ts::Index py_start = ((ith_chunk % xy_chunks) / l_x) * shape_vec[3];

                // ts::Index nx_last = shape[3] % shape_vec[3];
                // ts::Index ny_last = shape[4] % shape_vec[4];
                ts::Index nx_last = shape[2] % shape_vec[2];
                ts::Index ny_last = shape[3] % shape_vec[3];
                // ts::Index my_nx = (ith_chunk % xy_chunks) % l_x == l_x - 1 ? nx_last : shape_vec[3];
                // ts::Index my_ny = (ith_chunk % xy_chunks) / l_x == l_y - 1 ? ny_last : shape_vec[4];
                ts::Index my_nx = (ith_chunk % xy_chunks) % l_x == l_x - 1 ? nx_last : shape_vec[2];
                ts::Index my_ny = (ith_chunk % xy_chunks) / l_x == l_y - 1 ? ny_last : shape_vec[3];

                ts::Index chunk_elements = my_channels * my_nx * my_ny;

                std::vector<float> chunk_data_ch1(chunk_elements);
                // std::vector<float> chunk_data_ch1(my_nx * my_ny);

                // For using TensorStore to open the uncompressed zarr
                // compressor = true;

                if (compressor == true) {
                    nlohmann::json chunk_spec = {
                        {"driver", "zarr"},
                        {"kvstore", {{"driver", "file"}, {"path", zarr_path}}},
                        {"metadata", {{"chunks", {1, shape_vec[1], shape_vec[2], shape_vec[3]}},
                                      //{"compressor", nullptr},
                                      {"dtype", "<f4"},
                                      {"fill_value", "NaN"},
                                      {"filters", nullptr},
                                      {"order", "C"},
                                      {"shape", {1, shape[1], shape[2], shape[3]}},
                                      {"zarr_format", 2}}},
                        {"dtype", "float32"},
                        {"transform", {{"input_shape", {shape_vec[0], my_channels, my_nx, my_ny}}, {"output", {
                                                                                                              {{"input_dimension", 0}, {"offset", 0}, {"stride", 1}},
                                                                                                              {{"input_dimension", 1}, {"offset", ch_start}, {"stride", 1}}, // 64 channel
                                                                                                              //{{"input_dimension", 1}, {"offset", 0}, {"stride", 1}},
                                                                                                              {{"input_dimension", 2}, {"offset", px_start}, {"stride", 1}},
                                                                                                              {{"input_dimension", 3}, {"offset", py_start}, {"stride", 1}},
                                                                                                          }}}}};

                    auto result = ts::Open<float, zarr_dim>(chunk_spec, ts::Context::Default()).result();

                    if (!result.ok()) {
                        std::cerr << "Open zarr failed: " << result.status() << "\n";
                    }

                    auto read_result = ts::Read(*result).result();
                    if (!read_result.ok()) {
                        std::cerr << "Read failed: " << read_result.status() << "\n";
                    }
                    auto array_chunk = *read_result;
                    float *ptr = array_chunk.data();
                    std::copy(ptr, ptr + chunk_elements, chunk_data_ch1.begin());
                } else {
                    file_num[0] = 0;
                    file_num[1] = (ith_chunk / xy_chunks);
                    file_num[2] = (ith_chunk % xy_chunks) % l_x;
                    file_num[3] = (ith_chunk % xy_chunks) / l_x;

                    file_num_string = std::to_string(file_num[0]) +
                                  '.' + std::to_string(file_num[1]) +
                                  '.' + std::to_string(file_num[2]) +
                                  '.' + std::to_string(file_num[3]);

                    std::ifstream file(zarr_path + '/' + file_num_string, std::ios::binary);

                    if (!file) {
                        std::cerr << "No file named: " << zarr_path << '/' << file_num_string << std::endl;
                    }

                    if (my_nx == shape_vec[2] && my_ny == shape_vec[3]) {
                        if (!file.read(reinterpret_cast<char *>(chunk_data_ch1.data()), chunk_elements * sizeof(float))) {
                            std::cerr << "Read failed!" << std::endl;
                        }
                    } else {
                        // Preparing a full-size array to read the chunk including null values
                        // std::vector<float> chunk_data_full(shape_vec[2] * shape_vec[3], -10000.0f);
                        // 64 channels (spectrum)
                        std::vector<float> chunk_data_full(shape_vec[1] * shape_vec[2] * shape_vec[3], -10000.0f);
                        if (!file.read(reinterpret_cast<char *>(chunk_data_full.data()), shape_vec[1] * shape_vec[2] * shape_vec[3] * sizeof(float))) {
                            std::cerr << "Read failed!" << std::endl;
                        }

                        // resize
                        ts::Index idx = 0;
                        // for (ts::Index i = 0; i < shape_vec[2] * shape_vec[3]; ++i) {
                        // 64 channels (spectrum)
                        for (ts::Index i = 0; i < shape_vec[1] * shape_vec[2] * shape_vec[3]; ++i) {
                            float full_val = chunk_data_full[i];
                            if (!std::isnan(full_val))
                            {
                                chunk_data_ch1[idx] = full_val;
                                idx++;
                            }
                        }
                        chunk_data_full.clear();
                    }
                }

                auto io_end = std::chrono::high_resolution_clock::now();

                io_time += std::chrono::duration<double>(io_end - io_start).count();

                ////////// Don't wait. Output spectrum when the 1st-round chunks are finished.

                // Time computation
                auto compute_start = std::chrono::high_resolution_clock::now();

                // 64 channels (spectrum)
                for (ts::Index i = 0; i < chunk_elements; ++i) {
                    float value = chunk_data_ch1[i];

                    if (!std::isnan(value)) {
                        long channel_index = (ch_start + i / (my_nx * my_ny));
                        // local_spectrum[channel_index] += value;
                        // valid_pixels[channel_index] += 1;
                        thread_spectrum[channel_index] += value;
                        thread_valid[channel_index] += 1;
                    }
                }
                chunk_data_ch1 = {};

                auto compute_end = std::chrono::high_resolution_clock::now();

                compute_time += std::chrono::duration<double>(compute_end - compute_start).count();
            }
            #pragma omp critical
            {
                for (long c = 0; c < n_channels; ++c) {
                    local_spectrum[c] += thread_spectrum[c];
                    valid_pixels[c] += thread_valid[c];
                }
            }
        }
    }

    // Time computation
    auto compute_start = std::chrono::high_resolution_clock::now();

    // Normalize the local spectrum
    for (long c = 0; c < n_channels; ++c) {
        if (valid_pixels[c] > 0) {
            local_spectrum[c] /= valid_pixels[c];
        }
    }

    // Output spectrum
    std::ofstream specout("spectrum_nonmpi.dat");
    for (long c = 0; c < n_channels; ++c) {
        specout << c << "\t" << local_spectrum[c] << "\n";
    }
    specout.close();

    auto compute_end = std::chrono::high_resolution_clock::now();
    compute_time += std::chrono::duration<double>(compute_end - compute_start).count();

    auto total_end = std::chrono::high_resolution_clock::now();
    total_time += std::chrono::duration<double>(total_end - total_start).count();

    std::cout << "I/O time: " << io_time << " seconds\n";
    std::cout << "Compute time: " << compute_time << " seconds\n";
    std::cout << "Total time: " << total_time << " seconds\n";
    outfile << total_time << " " << io_time << " " << compute_time << " sec." << std::endl;

    return 0;
}
