#include <chrono>
#include <filesystem>
#include <fmt/format.h>
#include <sndfile.hh>

namespace chrono = std::chrono;
namespace fs = std::filesystem;

#include "SNAFxModel.h"

void write_file(const fs::path& file_path, const std::vector<float>& data, int sample_rate)
{
    fmt::print("Writing output to file: {}\n", file_path.string());
    SndfileHandle file { file_path.c_str(), SFM_WRITE, SF_FORMAT_WAV | SF_FORMAT_PCM_16, 1, sample_rate };
    file.write(data.data(), (sf_count_t)data.size());
}

std::pair<std::vector<float>, int> read_file(const fs::path& file_path)
{
    fmt::print("Reading from input: {}\n", file_path.string());
    SndfileHandle file { file_path.c_str() };

    fmt::print("    File sample rate: {}\n", file.samplerate());
    fmt::print("    File # channels: {}\n", file.channels());
    fmt::print("    File # samples: {}\n", file.frames());

    if(file.channels() != 1)
        return {};

    std::vector<float> data;
    data.resize(std::min((size_t)file.frames(), size_t(10.0 * file.samplerate())), 0.0);
    file.read(data.data(), (sf_count_t)data.size());
    return { data, (int)file.samplerate() };
}

int main()
{
    const auto model_file_path = fs::path { std::string { RTNEURAL_EXPERIMENTS_SOURCE_DIR } + std::string { "/examples/snafx/model_comp.json" } };

    std::ifstream json_stream(model_file_path, std::ifstream::binary);
    const auto model_json = nlohmann::json::parse(json_stream);
    //    fmt::print ("{}\n", model_json.dump());

    snafx::Model<float, 32, 9, 2> model;
    model.load_model(model_json);
    model.reset();
    model.condition({ 0.0f, 0.0f });

    const auto [test_ins, sample_rate] = read_file("test.wav");
    const auto num_samples = test_ins.size();
    std::vector<float> test_outs(num_samples, 0.0f);

    const auto start_time = chrono::high_resolution_clock::now();

    for(size_t i = 0; i < num_samples; ++i)
        test_outs[i] = 0.25f * model.forward(test_ins[i]);

    const auto stop_time = chrono::high_resolution_clock::now();
    const auto duration = chrono::duration_cast<chrono::milliseconds>(stop_time - start_time);
    const auto seconds_processed = (float)num_samples / (float)sample_rate;
    fmt::print("Processed {} seconds of audio at {} sample rate in {} milliseconds\n",
        seconds_processed,
        sample_rate,
        duration.count());
    fmt::print("Speed: {:.4f}x real-time\n", seconds_processed / (0.001 * (double)duration.count()));

    write_file("test_out.wav", test_outs, sample_rate);

    return 0;
}
