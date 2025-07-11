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
#include <mpi.h>


namespace ts = tensorstore;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    bool histogram = true;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    //const std::string zarr_path = "alma16G.zarr";
    //const std::string zarr_path = "askap_hydra_extragalactic_128_v2.zarr/SKY";
    const std::string zarr_path = "as_128_3D.zarr/data";

    // open binary directly
    if (rank == 0) {
        const std::string filename = "as_128_3D.zarr/data/0.0.0.0";
        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            std::cerr << "No file named: " << filename << std::endl;
            return 1;
        }
        file.seekg(0, std::ios::end);
        std::streamsize filesize = file.tellg();
        file.seekg(0, std::ios::beg);

        if (filesize % sizeof(float) != 0) {
            std::cerr << "Data is not float!" << std::endl;
            return 1;
        }
        std::vector<float> first_data(filesize / sizeof(float));
        if (!file.read(reinterpret_cast<char*>(first_data.data()), filesize)) {
            std::cerr << "Read failed!" << std::endl;
            return 1;
        }
        for (size_t i = 0; i < std::min(first_data.size(), size_t(20)); ++i) {
            std::cout << "data[" << i << "] = " << first_data[i] << std::endl;
        }

    }

    constexpr int zarr_dim = 4;
    //constexpr int zarr_dim = 5;
    std::ofstream outfile("result" + std::to_string(size) + "_12G.dat", std::ios::app);
    
    const int bitpix = 4;  // float32 = 4 bytes

    ts::Index npixels = 0, n_channels = 0, n_x = 0, n_y = 0;
    ts::Index less_pixels = 0, less_channels = 0;

    std::vector<ts::Index> shape_vec(zarr_dim, 0);
    //int chunk_size = 1024;
    // 64 channels
    int chunk_size = 16;
    //int chunk_size = 1;

    double io_open_time = 0.0;
    double io_read_time = 0.0;
    double compute_time = 0.0;
    double total_time = 0.0;
    MPI_Bcast(&io_open_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&io_read_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&compute_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&total_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    //MPI_Barrier(MPI_COMM_WORLD);
    auto total_start = std::chrono::high_resolution_clock::now();

    //MPI_Barrier(MPI_COMM_WORLD); // Synchronize all processes before starting I/O
    auto io_open_start = std::chrono::high_resolution_clock::now();
    std::vector<ts::Index> shape(zarr_dim, 0);

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
        //shape = store.domain().shape();
        auto shape_span = store.domain().shape();
        shape.assign(shape_span.begin(), shape_span.end());

        n_channels = shape[1];
        n_x = shape[zarr_dim-2];
        n_y = shape[zarr_dim-1];
        std::cout << "Channels: " << n_channels << ", X: " << n_x << ", Y: " << n_y << "\n";
        npixels = ts::Index(n_channels) * n_x * n_y;

        //std::cout << "Chunk size: " << chunk_size << "\n";
        if (zarr_dim == 5) {
            shape_vec = {shape[0], chunk_size, shape[2], 256, 256};
        } else if (zarr_dim == 4) {
            shape_vec = {shape[0], chunk_size, 256, 256};
        } else {
            std::cerr << "Unsupported Zarr dimension: " << zarr_dim << "\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        // ts::Index mem_pixels = size * chunk_size * n_x * n_y;
        // if (npixels > mem_pixels) {
        //     less_pixels = mem_pixels;
        //     less_channels = size * chunk_size;
        // } else {
        //     less_pixels = npixels;
        //     less_channels = n_channels;
        // }
        // clear store
        store = {};
        //result_shape = {};
    }
    //std::cout << "less_channels: " << less_channels << "\n";

    // Broadcast shared metadata
    //MPI_Bcast(shape_vec.data(), 4, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(shape.data(), zarr_dim, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(shape_vec.data(), zarr_dim, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&npixels, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n_channels, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n_x, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n_y, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&less_pixels, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&less_channels, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
    
    std::cout << "shape: ";
    for (ts::Index i = 0; i < shape.size(); ++i) {
        std::cout << shape[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "shape_vec: ";
    for (ts::Index i = 0; i < shape_vec.size(); ++i) {
        std::cout << shape_vec[i] << " ";
    }
    std::cout << std::endl;
    //auto l_x = shape[3] / shape_vec[3] + ((shape[3] % shape_vec[3]) ? 1 : 0); // 30 + 1
    //auto l_y = shape[4] / shape_vec[4] + ((shape[4] % shape_vec[4]) ? 1 : 0); // 18 + 1
    auto l_x = shape[2] / shape_vec[2] + ((shape[2] % shape_vec[2]) ? 1 : 0); // 30 + 1
    auto l_y = shape[3] / shape_vec[3] + ((shape[3] % shape_vec[3]) ? 1 : 0); // 18 + 1
    // 64 channels
    //auto l_z = shape[1] / shape_vec[1] + ((shape[1] % shape_vec[1]) ? 1 : 0); // 2
    auto l_z = 1;
    //std::cout << "l_x: " << l_x << ", l_y: " << l_y << ", l_z: " << l_z << "\n";

    
    //ts::Index total_chunks = l_x * l_y * l_z; // 31 * 19 * 2 = 1178
    ts::Index total_chunks = l_x * l_y * 1; // 31 * 19
    //std::cout << "total chunks = " << total_chunks << std::endl;
    ts::Index total_loops = total_chunks / size + ((total_chunks % size) ? 1 : 0); // 1778 / 10 + 1 = 178
    


    // Initialize local and global histograms
    std::vector<long long int> local_histogram(total_chunks * 256, 0);
    std::vector<long long int> global_histogram(total_chunks * 256, 0);

    int xy_chunks = l_x * l_y; // 31 * 19 = 589

    bool no_chunk = false;
    ts::Index my_channels = 1;
    std::vector<float> max_values(total_chunks, -1);
    std::vector<float> min_values(total_chunks, 1);

    //MPI_Barrier(MPI_COMM_WORLD); // Synchronize all processes after I/O
    auto io_open_end = std::chrono::high_resolution_clock::now();
    if (rank == 0) {
        io_open_time += std::chrono::duration<double>(io_open_end - io_open_start).count();
    }

    for (ts::Index nloop = 0; nloop < total_loops; ++nloop) {
        if (rank == 0) {
            std::cout << "Processing loop " << nloop + 1 << " of " << total_loops << "\n";
        }
        // Time I/O
        //MPI_Barrier(MPI_COMM_WORLD); // Synchronize all processes before starting I/O
        auto io_open_start = std::chrono::high_resolution_clock::now();

        // Each rank read a chunk from ch_start to ch_start + ch_len
        // 64 channels
        //my_channels = (nloop == total_loops - 1) ? shape[1] % shape_vec[1] : shape_vec[1];
        
        
        int ith_chunk = nloop * size + rank;
        if (ith_chunk >= total_chunks) {
            no_chunk = true;
            my_channels = 0;
        }
        ts::Index ch_start = (ith_chunk / xy_chunks) * shape_vec[1];

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
            //std::cout << nloop << " " << chunk_elements << " " << my_nx << " " << my_ny << std::endl;

            //std::cout << ith_chunk << " " << ch_start << " " << px_start << " " << py_start << std::endl;
            //std::cout << l_x << " " << l_y << " " << xy_chunks << " " << std::endl;
            //std::cout << my_nx << " " << my_ny << " " << my_channels << std::endl;

            //std::cout << "ith_chunk: " << ith_chunk << ", my_nx: " << my_nx << ", my_ny: " << my_ny << "\n";

            std::vector<float> chunk_data(chunk_elements);
        
            nlohmann::json chunk_spec = {
                {"driver", "zarr"},
                {"kvstore", {
                    {"driver", "file"},
                    {"path", zarr_path}
                }},
                {"metadata", {
                    {"chunks", {1, shape_vec[1], shape_vec[2], shape_vec[3]}},
                    {"compressor", nullptr},
                    {"dtype", "<f4"},
                    {"fill_value", 0.0},
                    {"filters", nullptr},
                    {"order", "C"},
                    {"shape", {1, shape[1], shape[2], shape[3]}},
                    {"zarr_format", 2}
                }},
                {"dtype", "float32"},
                {"transform", { 
                    {"input_shape", {shape_vec[0], my_channels, my_nx, my_ny}},
                    {"output", {
                        {{"input_dimension", 0}, {"offset", 0}, {"stride", 1}},
                        //{{"input_dimension", 1}, {"offset", ch_start}, {"stride", 1}}, // 64 channel
                        {{"input_dimension", 1}, {"offset", 0}, {"stride", 1}},
                        {{"input_dimension", 2}, {"offset", px_start}, {"stride", 1}},
                        {{"input_dimension", 3}, {"offset", py_start}, {"stride", 1}},
                    }}
                }}
            };
            //auto result = ts::Open<float, 4>(chunk_spec, ts::Context::Default()).result();
            auto result = ts::Open<float, zarr_dim>(chunk_spec, ts::Context::Default()).result();
            //auto result = Open(chunk_spec, ts::ReadWriteMode::read).result();
            //auto result = ts::Open<ts::shared_array<void>, zarr_dim>(chunk_spec, context).result();
            
            if (!result.ok()) {
                std::cerr << "Open zarr failed: " << result.status() << "\n";
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            //MPI_Barrier(MPI_COMM_WORLD);
            auto io_open_end = std::chrono::high_resolution_clock::now();
            if (rank == 0) {
                io_open_time += std::chrono::duration<double>(io_open_end - io_open_start).count();
            }
            //MPI_Barrier(MPI_COMM_WORLD); // Synchronize all processes before starting I/O
            auto io_read_start = std::chrono::high_resolution_clock::now();

            auto read_result = ts::Read(*result).result();
            if (!read_result.ok()) {
                std::cerr << "Read failed: " << read_result.status() << "\n";
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            

            //tensorstore::SharedArray<float, 4> array = tensorstore::Cast<float, 4>(*read_result);
            //ts::SharedArray<float, 4> array = static_cast<ts::SharedArray<float, 4>>(*read_result);

            // if open with ts::ReadWriteMode::read
            // auto& arr = *read_result;
            // if (!arr.data()) {
            //     std::cerr << "arr.data() is nullptr! Possibly invalid read.\n";
            //     MPI_Abort(MPI_COMM_WORLD, 1);
            // }
            // float* array = static_cast<float*>(arr.data());

            //std::cout << ith_chunk << " shape = ";
            //for (auto s : arr.shape()) std::cout << s << " ";
            //std::cout << std::endl;          
            
            //auto array_chunk = *read_result;
            //float* ptr = array_chunk.data();

            // try not to use new array, use tensorStore
            //std::copy(ptr, ptr + chunk_elements, chunk_data.begin());

            //std::cout << (*read_result).data()[0] << std::endl;
            //std::cout << array_chunk.data()[0]<< std::endl;
            //std::cout << ptr[0] << std::endl;
            //std::cout << chunk_data[0] << std::endl;

            // clear temp array
            //array_chunk = {};
        
            //MPI_Barrier(MPI_COMM_WORLD); // Synchronize all processes after I/O
            auto io_read_end = std::chrono::high_resolution_clock::now();
            if (rank == 0) {
                io_read_time += std::chrono::duration<double>(io_read_end - io_read_start).count();
            }

            if (ith_chunk == 0) {
                for (int i = 0; i < 20; ++i) {
                    std::cout << read_result->data()[i] << std::endl;
                }
            }

            // Time computation
            //MPI_Barrier(MPI_COMM_WORLD);
            auto compute_start = std::chrono::high_resolution_clock::now();

            for (ts::Index i = 0;  i < my_nx * my_ny; ++i) {
                //auto ith_element = chunk_data[i];
                //auto ith_element = array_chunk.data()[i];
                auto ith_element = read_result->data()[i];
                
                //float ith_element = array[i];
                
                if (!std::isnan(ith_element)) {
                    //if (ith_chunk == 0) std::cout << ith_element << std::endl;
                    if (ith_element < min_values[ith_chunk]){
                        min_values[ith_chunk] = ith_element;
                        //std::cout << min_values[ith_chunk] << " " << ith_chunk << std::endl;
                    }
                    if (ith_element > max_values[ith_chunk]){
                        max_values[ith_chunk] = ith_element;
                        //std::cout << max_values[ith_chunk] << " " << ith_chunk << std::endl;
                    }
                } else {
                    //std::cout << ith_chunk << " " << ith_element << " " << ch_start << " " << px_start << " " << py_start << std::endl;
                }
            }


            // if (min_values[ith_chunk] == 1) {
            //     min_values[ith_chunk] = 0;
            // }
            // if (max_values[ith_chunk] == -1) {
            //     max_values[ith_chunk] = 0;
            // }
            //std::cout << max_values[ith_chunk] << " " << min_values[ith_chunk] << std::endl;
            
        
            for (ts::Index i = 0; i < my_nx * my_ny; ++i) {
                //float value = chunk_data[i];
                //float value = array_chunk.data()[i];
                float value = read_result->data()[i];
                //float value = array[i];

                if (!std::isnan(value)) {
                    //relative_value =  255 * (value - min[ith_chunk]) / (*max - min[ith_chunk]);
                    if (max_values[ith_chunk] - min_values[ith_chunk] != 0) { 
                        auto relative_value = static_cast<int>(255 * (value - min_values[ith_chunk]) / (max_values[ith_chunk] - min_values[ith_chunk]));
                        local_histogram[ith_chunk * 256 + relative_value] += 1;
                    }
                } else {
                    //std::cout << "nan: " << i << " " << my_nx << " " << my_nx << std::endl;
                    //local_histogram[ith_chunk * 256 + 0] += 1;
                }
            }
            //array_chunk = {};
            //array = {};

            //std::cout << nloop << ": Rank " << rank << " finished processing chunk with " << ch_len << " channels.\n";
            //chunk_data.clear();
            //chunk_data.shrink_to_fit();

            //MPI_Barrier(MPI_COMM_WORLD);
            auto compute_end = std::chrono::high_resolution_clock::now();
            if (rank == 0) {
                compute_time += std::chrono::duration<double>(compute_end - compute_start).count();
            }
            
        }
        //MPI_Barrier(MPI_COMM_WORLD);
    }

    //std::cout << "\nLoop ends.\n" << std::endl;
    

    // Gather histogram
    MPI_Reduce(local_histogram.data(), global_histogram.data(), total_chunks*256, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    std::vector<float> global_max_values(total_chunks, -1.0f);
    std::vector<float> global_min_values(total_chunks, 1.0f);
    MPI_Reduce(max_values.data(), global_max_values.data(), total_chunks, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(min_values.data(), global_min_values.data(), total_chunks, MPI_FLOAT, MPI_MIN, 0, MPI_COMM_WORLD);

    auto final_histogram = std::vector<float>(256, 0.0f);
    // Time computation
    //MPI_Barrier(MPI_COMM_WORLD);
    auto compute_start = std::chrono::high_resolution_clock::now();
    if (rank == 0) {
        // for (int i; i < total_chunks; ++i) {
        //     std::cout << global_max_values[i] << std::endl;
        // }
        // final min and max
        float final_max = -1;
        float final_min = 1;
        for (ts::Index i = 0;  i < total_chunks; ++i) {
            if (!std::isnan(global_max_values[i]) && global_max_values[i] > final_max) {
                final_max = global_max_values[i];
                //std::cout << final_max << std::endl;
            }
            if (!std::isnan(global_min_values[i]) && global_min_values[i] < final_min) {
                final_min = global_min_values[i];
                //std::cout << final_min << std::endl;
            }
        }
        //auto final_max = std::max_element(global_max_values.begin(), global_max_values.end());
        //auto final_min = std::min_element(global_min_values.begin(), global_min_values.end());
        std::cout << "Final max: " << final_max << ", Final min: " << final_min << "\n";

        auto final_range = final_max - final_min;
        std::cout << "Final range: " << final_range << "\n";
    
        // for (ts::Index i_chunk = 0; i_chunk < total_chunks; ++i_chunk) {
        //     for (ts::Index i_value = 0; i_value < 256; ++i_value) {
        //         // Accumulate the histogram values
        //         //std::cout << "i_chunk range: " << (global_max_values[i_chunk] - global_min_values[i_chunk]) << std::endl;
        //         if (global_max_values[i_chunk] - global_min_values[i_chunk] != 0) {
        //             ts::Index final_index = int(i_value * final_range / (global_max_values[i_chunk] - global_min_values[i_chunk]));
        //             final_histogram[final_index] += global_histogram[i_chunk * 256 + i_value];
        //         }
        //     }
        // }

        std::ofstream output("histogram_12G.dat");
        // for (ts::Index i_value = 0; i_value < 256; ++i_value) {
        //     output << i_value << " " << final_histogram[i_value] << "\n";
        // }
        for (ts::Index i_value = 0; i_value < 256; ++i_value) {
            //for (ts::Index i_chunk = 0; i_chunk < total_chunks; ++i_chunk) {
            for (ts::Index i_chunk = 0; i_chunk < total_chunks; ++i_chunk) {
                if (global_max_values[i_chunk] - global_min_values[i_chunk] != 0) {
                    ts::Index final_index = int(i_value * final_range / (global_max_values[i_chunk] - global_min_values[i_chunk]));
                    final_histogram[final_index] += global_histogram[i_chunk * 256 + i_value];
                }
            }
            output << i_value << " " << final_histogram[i_value] << "\n";
        }
    }


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
        std::cout << "I/O open time: " << io_open_time << " seconds\n";
        std::cout << "I/O read time: " << io_read_time << " seconds\n";
        std::cout << "Compute time: " << compute_time << " seconds\n";
        std::cout << "Total time: " << total_time << " seconds\n";
        outfile << total_time << " " << io_open_time + io_read_time << " " << compute_time << " sec." << std::endl;
    }

    MPI_Finalize();

    return 0;
}
