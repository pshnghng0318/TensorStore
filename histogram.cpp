#include <tensorstore/tensorstore.h>
#include <tensorstore/open.h>
#include <tensorstore/util/result.h>
#include <tensorstore/context.h>
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <cmath>
#include <mpi.h>

namespace ts = tensorstore;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    //const std::string zarr_path = "alma16G.zarr";
    const std::string zarr_path = "askap_hydra_extragalactic_128.zarr/SKY";
    constexpr int zarr_dim = 5;
    std::ofstream outfile("result" + std::to_string(size) + "_12G.dat", std::ios::app);
    
    const int bitpix = 4;  // float32 = 4 bytes

    ts::Index npixels = 0, n_channels = 0, n_x = 0, n_y = 0;
    ts::Index less_pixels = 0, less_channels = 0;

    std::vector<ts::Index> shape_vec(zarr_dim, 0);
    //int chunk_size = 1024;
    int chunk_size = 64;

    double io_time = 0.0;
    double compute_time = 0.0;
    double total_time = 0.0;
    MPI_Bcast(&io_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&compute_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&total_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    //MPI_Barrier(MPI_COMM_WORLD);
    auto total_start = std::chrono::high_resolution_clock::now();

    //MPI_Barrier(MPI_COMM_WORLD); // Synchronize all processes before starting I/O
    auto io_start = std::chrono::high_resolution_clock::now();

    if (rank == 0) {
        // Replace the 1st number with your Zarr file settings
        
        nlohmann::json spec = {
            {"driver", "zarr"},
            {"kvstore", {
                {"driver", "file"},
                {"path", zarr_path}
            }},
            {"dtype", "float32"},
        };

        nlohmann::json context_spec = {
            {"cache_pool", {
                {"total_bytes_limit", 256 << 20}
            }}
        };
        auto context_result = ts::Context::FromJson(context_spec);
        if (!context_result.ok()) {
            std::cerr << "Failed to create context: " << context_result.status() << "\n";
            return 1;
        }
        auto context = *context_result;

        //auto result_shape = ts::Open<float, 4>(spec, context).result();
        auto result_shape = ts::Open<float, zarr_dim>(spec, context).result();
        if (!result_shape.ok()) {
            std::cerr << "Open failed: " << result_shape.status() << "\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        auto store = *result_shape;
        auto shape = store.domain().shape();
        //std::cout << "Shape: " << shape << "\n";

        n_channels = shape[1];
        n_x = shape[zarr_dim-2];
        n_y = shape[zarr_dim-1];
        std::cout << "Channels: " << n_channels << ", X: " << n_x << ", Y: " << n_y << "\n";
        npixels = ts::Index(n_channels) * n_x * n_y;

        //std::cout << "Chunk size: " << chunk_size << "\n";
        if (zarr_dim == 5) {
            shape_vec = {shape[0], chunk_size, shape[2], shape[3], shape[4]};
        } else if (zarr_dim == 4) {
            shape_vec = {shape[0], chunk_size, shape[2], shape[3]};
        } else {
            std::cerr << "Unsupported Zarr dimension: " << zarr_dim << "\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        ts::Index mem_pixels = size * chunk_size * n_x * n_y;
        if (npixels > mem_pixels) {
            less_pixels = mem_pixels;
            less_channels = size * chunk_size;
        } else {
            less_pixels = npixels;
            less_channels = n_channels;
        }
        store = {};
        //result_shape = {};
    }
    //std::cout << "less_channels: " << less_channels << "\n";

    // Broadcast shared metadata
    //MPI_Bcast(shape_vec.data(), 4, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(shape_vec.data(), zarr_dim, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&npixels, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n_channels, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n_x, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n_y, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&less_pixels, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&less_channels, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);

    //std::vector<float> local_spectrum(n_channels, 0.0f);
    //std::vector<int> local_valid(n_channels, 0);
    //std::vector<float> global_spectrum(n_channels, 0.0f);
    //std::vector<int> global_valid(n_channels, 0);
    std::vector<float> local_histogram(255, 0.0f);
    std::vector<float> global_histogram(255, 0.0f);

    int l_x = shape[3] / shape_vec[3] + ((shape[3] % shape_vec[3]) ? 1 : 0); // 30 + 1
    int l_y = shape[4] / shape_vec[4] + ((shape[4] % shape_vec[4]) ? 1 : 0); // 18 + 1
    int l_z = shape[1] / shape_vec[1] + ((shape[1] % shape_vec[1]) ? 1 : 0); // 2
    std::cout << "l_x: " << l_x << ", l_y: " << l_y << ", l_z: " << l_z << "\n";
    
    ts::Index total_chunks = l_x * l_y * l_z; // 31 * 19 * 2 = 1178
    //ts::Index total_loops = (npixels / less_pixels) + ((npixels % less_pixels) ? 1 : 0);
    ts::Index total_loops = total_chunks / size + ((total_chunks % size) ? 1 : 0); // 1778 / 10 + 1 = 178

    int xy_chunks = l_x * l_y; // 31 * 19 = 589

    bool no_chunks = false;
    ts::Index my_channels = chunk_size;

    //MPI_Barrier(MPI_COMM_WORLD); // Synchronize all processes after I/O
    auto io_end = std::chrono::high_resolution_clock::now();
    if (rank == 0) {
        io_time += std::chrono::duration<double>(io_end - io_start).count();
    }

    for (ts::Index nloop = 0; nloop < total_loops; ++nloop) {
        if (rank == 0) {
            std::cout << "Processing loop " << nloop + 1 << " of " << total_loops << "\n";
        }
        // Time I/O
        //MPI_Barrier(MPI_COMM_WORLD); // Synchronize all processes before starting I/O
        auto io_start = std::chrono::high_resolution_clock::now();

        // Each rank read a chunk from ch_start to ch_start + ch_len
        //ts::Index ch_start = nloop * less_channels + (rank * chunk_size);
        ts::Index ch_start = nloop * shape_vec[1];
        
        // if (n_channels - nloop * less_channels - rank * chunk_size > chunk_size) {
        //     my_channels = chunk_size;
        // } else if (n_channels - nloop * less_channels - rank * chunk_size > 0) {
        //     my_channels = n_channels - nloop * less_channels - rank * chunk_size;
        // } else{
        //     no_chunks = true;
        // }
        my_channels = nloop == total_loops - 1 ? shape[1] % shape_vec[1] : shape_vec[1];
        std::cout << nloop << " : Rank " << rank << " processing " << my_channels << " channels from " << ch_start << "\n";
        int ith_chunk = nloop * size + rank;
        if (ith_chunk >= total_chunks) {
            no_chunks = true;
            my_channels = 0;
        }
        ts::Index px_start = ((ith_chunk % xy_chunks) % l_x - 1) * shape_vec[3];
        ts::Index py_start = ((ith_chunk % xy_chunks) / l_x - 1) * shape_vec[4];

        //ts::Index chunk_elements = my_channels * n_x * n_y;
        ts::Index last_chunks = ith_chunk % xy_chunks;
        ts::Index nx_last = shape[3] % shape_vec[3];
        ts::Index ny_last = shape[4] % shape_vec[4];
        ts::Index my_nx = (ith_chunk % xy_chunks) % l_x == l_x - 1 ? nx_last : shape_vec[3];
        ts::Index my_ny = (ith_chunk % xy_chunks) / l_x == l_y - 1 ? ny_last : shape_vec[4];
        std::cout << "ith_chunk: " << ith_chunk << ", my_nx: " << my_nx << ", my_ny: " << my_ny << "\n";
        ts::Index chunk_elements = my_channels * my_nx * my_ny;

        std::vector<float> chunk_data(chunk_elements);
        if (!no_chunks) {
            nlohmann::json chunk_spec = {
                {"driver", "zarr"},
                {"kvstore", {
                    {"driver", "file"},
                    {"path", zarr_path}
                }},
                {"dtype", "float32"},
                {"transform", {
                    //{"input_shape", {shape_vec[0], my_channels, shape_vec[2], shape_vec[3]}},
                    {"input_shape", {shape_vec[0], my_channels, shape_vec[2], my_nx, my_ny}},
                    {"output", {
                        //{{"input_dimension", 0}, {"offset", 0}, {"stride", 1}},
                        //{{"input_dimension", 1}, {"offset", ch_start}, {"stride", 1}},
                        //{{"input_dimension", 2}, {"offset", 0}, {"stride", 1}},
                        //{{"input_dimension", 3}, {"offset", 0}, {"stride", 1}},
                        {{"input_dimension", 0}, {"offset", 0}, {"stride", 1}},
                        {{"input_dimension", 1}, {"offset", ch_start}, {"stride", 1}},
                        {{"input_dimension", 2}, {"offset", 0}, {"stride", 1}},
                        {{"input_dimension", 3}, {"offset", px_start}, {"stride", 1}},
                        {{"input_dimension", 4}, {"offset", py_start}, {"stride", 1}},
                    }}
                }}
            };
            //auto result = ts::Open<float, 4>(chunk_spec, ts::Context::Default()).result();
            auto result = ts::Open<float, zarr_dim>(chunk_spec, ts::Context::Default()).result();
            if (!result.ok()) MPI_Abort(MPI_COMM_WORLD, 1);
            auto read_result = ts::Read(*result).result();
            if (!read_result.ok()) {
                std::cerr << "Read failed: " << read_result.status() << "\n";
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            auto array_chunk = *read_result;
            float* ptr = array_chunk.data();
            std::copy(ptr, ptr + chunk_elements, chunk_data.begin());
            array_chunk = {};
        }
        //MPI_Barrier(MPI_COMM_WORLD); // Synchronize all processes after I/O
        auto io_end = std::chrono::high_resolution_clock::now();
        if (rank == 0) {
            io_time += std::chrono::duration<double>(io_end - io_start).count();
        }

        // Time computation
        //MPI_Barrier(MPI_COMM_WORLD);
        auto compute_start = std::chrono::high_resolution_clock::now();

        // Divide the chunk data among ranks
        // for (ts::Index ch = 0; ch < my_channels; ++ch) {
        //     ts::Index global_ch = ch_start + ch;
        //     //if ((global_ch % size) != rank) continue;
        //     for (ts::Index i = 0; i < n_x * n_y; ++i) {
        //         float val = chunk_data[ch * n_x * n_y + i];
        //         if (!std::isnan(val)) {
        //             local_spectrum[global_ch] += val;
        //             local_valid[global_ch] += 1;
        //         }
        //     }
        //     if (local_valid[global_ch] > 0){
        //         local_spectrum[global_ch] /= local_valid[global_ch];
        //     }
        // }

        // Compute histogram        
        auto max_it[ith_chunk] = std::max_element(chunk_data.begin(), chunk_data.end());
        auto min_it[ith_chunk] = std::min_element(chunk_data.begin(), chunk_data.end());
        for (ts::Index ch = 0; ch < my_channels; ++ch) {
            //ts::Index global_ch = ch_start + ch;
            //if ((global_ch % size) != rank) continue;
            for (ts::Index i = 0; i < n_x * n_y; ++i) {
                float val = chunk_data[ch * n_x * n_y + i];
                if (!std::isnan(val)) {
                    // Update histogram
                    if (val >= 0 && val < n_channels) {
                        local_histogram[relative_value] += 1.0f;
                    }
                }
            }
        }

        //std::cout << nloop << ": Rank " << rank << " finished processing chunk with " << ch_len << " channels.\n";
        chunk_data.clear();
        chunk_data.shrink_to_fit();

        //MPI_Barrier(MPI_COMM_WORLD);
        auto compute_end = std::chrono::high_resolution_clock::now();
        if (rank == 0) {
            compute_time += std::chrono::duration<double>(compute_end - compute_start).count();
        }
    }

    // Time computation
    //MPI_Barrier(MPI_COMM_WORLD);
    auto compute_start = std::chrono::high_resolution_clock::now();

    // Gather spectrum
    //MPI_Reduce(local_spectrum.data(), global_spectrum.data(), n_channels, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Gather histogram
    MPI_Reduce(local_histogram.data(), global_histogram.data(), n_channels, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    

    //MPI_Barrier(MPI_COMM_WORLD);
    auto compute_end = std::chrono::high_resolution_clock::now();

    if (rank == 0) {
        compute_time += std::chrono::duration<double>(compute_end - compute_start).count();
        //std::cout << compute_time << std::endl;
    }
    //MPI_Barrier(MPI_COMM_WORLD);
    auto total_end = std::chrono::high_resolution_clock::now();
    if (rank == 0) {
        total_time += std::chrono::duration<double>(total_end - total_start).count();
        //std::cout << total_time << std::endl;
    }
    if (rank == 0) {
        std::cout << "I/O time: " << io_time << " seconds\n";
        std::cout << "Compute time: " << compute_time << " seconds\n";
        std::cout << "Total time: " << total_time << " seconds\n";
        outfile << total_time << " " << io_time << " " << compute_time << " sec." << std::endl;
    }

    if (rank == 0) {
        std::ofstream output("spectrum_12G.dat");
        for (ts::Index ch = 0; ch < n_channels; ++ch) {
            output << ch << " " << global_spectrum[ch] << "\n";
        }
    }

    MPI_Finalize();
    return 0;
}
