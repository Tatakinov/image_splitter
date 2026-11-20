#include <iostream>
#include <map>
#include <optional>
#include <queue>
#include <sstream>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

namespace {
    const int kMaxThreshold = 0xff * 12 + 1;
}

struct pos_t {
    int x, y;
};

struct spread_t {
    int x;
    int y;
    int index_region;
};

struct region_t {
    std::vector<bool> r;
    std::queue<pos_t> q;
};

template<class T>
T diff(const T &a, const T &b) {
    return std::max(a, b) - std::min(a, b);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        return 1;
    }
    double rate = 1.0;
    if (argc >= 3) {
        std::istringstream iss(argv[2]);
        iss >> rate;
        if (rate > 100) {
            rate = 100;
        }
        if (rate < 0) {
            rate = 0;
        }
    }
    int threshold = 0;
    if (argc >= 4) {
        std::istringstream iss(argv[3]);
        iss >> threshold;
        if (threshold > 0xff) {
            threshold = 0xff;
        }
        if (threshold < 0) {
            threshold = 0;
        }
    }
    int w, h, _;
    unsigned char *img = stbi_load(argv[1], &w, &h, &_, 4);
    if (img == nullptr) {
        return 1;
    }
    std::vector<unsigned char> src(w * h * 4);
    for (int i = 0; i < w * h * 4; i++) {
        src[i] = img[i];
    }
    stbi_image_free(img);
    std::vector<unsigned char> dst;
    dst.resize(src.size());
    try {
        Ort::Env env;
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
#if defined(_WIN32) || defined(WIN32)
        Ort::Session session(env, L"model.onnx", session_options);
#else
        Ort::Session session(env, "model.onnx", session_options);
#endif // WIN32
        std::array<int64_t, 4> input_shape = {4, 1, h, w};
        std::array<int64_t, 4> output_shape = {4, 1, h, w};
        std::vector<float> input;
        input.resize(src.size());
        std::vector<float> output;
        output.resize(dst.size());
        for (int i = 0; i < w * h; i++) {
            for (int c = 0; c < 4; c++) {
                input[c * w * h + i] = src[4 * i + c] / 255.0;
            }
        }
        auto mem_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(mem_info, input.data(), input.size(), input_shape.data(), input_shape.size());
        Ort::Value output_tensor = Ort::Value::CreateTensor<float>(mem_info, output.data(), output.size(), output_shape.data(), output_shape.size());
        const char *input_names[] = {"input"};
        const char *output_names[] = {"output"};
        Ort::RunOptions run_options;
        try {
            session.Run(run_options, input_names, &input_tensor, 1, output_names, &output_tensor, 1);
        }
        catch (Ort::Exception &e) {
            std::cout << e.what() << std::endl;
            return 1;
        }
        for (int i = 0; i < w * h; i++) {
            for (int c = 0; c < 4; c++) {
                int byte = std::round(output[c * w * h + i] * 255);
                dst[4 * i + c] = std::max(0, std::min(255, byte));
            }
        }
    }
    catch (Ort::Exception &e) {
        std::cout << e.what() << std::endl;
        return 1;
    }
    std::vector<int> blur;
    blur.resize(dst.size());
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int i = y * w + x;
            for (int c = 0; c < 4; c++) {
                int count = 0;
                if (y > 0) {
                    if (x > 0) {
                        blur[i * 4 + c] += dst[(i - w - 1) * 4 + c] * 1;
                        count += 1;
                    }
                    {
                        blur[i * 4 + c] += dst[(i - w) * 4 + c] * 2;
                        count += 2;
                    }
                    if (x + 1 < w) {
                        blur[i * 4 + c] += dst[(i - w + 1) * 4 + c] * 1;
                        count += 1;
                    }
                }
                {
                    if (x > 0) {
                        blur[i * 4 + c] += dst[(i - 1) * 4 + c] * 2;
                        count += 2;
                    }
                    {
                        blur[i * 4 + c] += dst[i * 4 + c] * 4;
                        count += 4;
                    }
                    if (x + 1 < w) {
                        blur[i * 4 + c] += dst[(i + 1) * 4 + c] * 2;
                        count += 2;
                    }
                }
                if (y + 1 < h) {
                    if (x > 0) {
                        blur[i * 4 + c] += dst[(i + w - 1) * 4 + c] * 1;
                        count += 1;
                    }
                    {
                        blur[i * 4 + c] += dst[(i + w) * 4 + c] * 2;
                        count += 2;
                    }
                    if (x + 1 < w) {
                        blur[i * 4 + c] += dst[(i + w + 1) * 4 + c] * 1;
                        count += 1;
                    }
                }
                blur[i * 4 + c] /= count;
            }
        }
    }
    for (int i = 0; i < w * h; i++) {
        dst[i * 4 + 0] = std::max(0, std::min(0xff, std::max(static_cast<int>(dst[i * 4 + 0]), blur[i * 4 + 0])));
        dst[i * 4 + 1] = std::max(0, std::min(0xff, std::max(static_cast<int>(dst[i * 4 + 1]), blur[i * 4 + 1])));
        dst[i * 4 + 2] = std::max(0, std::min(0xff, std::max(static_cast<int>(dst[i * 4 + 2]), blur[i * 4 + 2])));
        dst[i * 4 + 3] = std::max(0, std::min(0xff, std::max(static_cast<int>(dst[i * 4 + 3]), blur[i * 4 + 3])));
    }
    std::vector<bool> all;
    all.resize(w * h);
    auto cond = [&](int index) {
        return
            dst[index * 4 + 0] <= threshold &&
            dst[index * 4 + 1] <= threshold &&
            dst[index * 4 + 2] <= threshold &&
            dst[index * 4 + 3] <= threshold &&
            !all[index];
    };
#if 0
    std::vector<unsigned char> debug;
    debug.resize(blur.size());
    for (int i = 0; i < w * h; i++) {
        if (cond(i)) {
            debug[i * 4 + 0] = 0x00;
            debug[i * 4 + 1] = 0x00;
            debug[i * 4 + 2] = 0x00;
            debug[i * 4 + 3] = 0xff;
        }
        else {
            debug[i * 4 + 0] = dst[i * 4 + 0];
            debug[i * 4 + 1] = dst[i * 4 + 1];
            debug[i * 4 + 2] = dst[i * 4 + 2];
            debug[i * 4 + 3] = 0xff;
        }
    }
    stbi_write_png("output.png", w, h, 4, debug.data(), w * 4);
#endif
    //std::cout << "onnx " << w << "," << h << std::endl;
    std::vector<region_t> parts;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int index = (y * w + x);
            if (cond(index)) {
                region_t part;
                part.r.resize(all.size());
                std::queue<pos_t> q;
                q.push({x, y});
                while (!q.empty()) {
                    auto [x, y] = q.front();
                    q.pop();
                    if (!cond(y * w + x)) {
                        continue;
                    }
                    all[y * w + x] = true;
                    part.r[y * w + x] = true;
                    if (x > 0 && cond(y * w + (x - 1))) {
                        q.push({x - 1, y});
                    }
                    if (x + 1 < w && cond(y * w + (x + 1))) {
                        q.push({x + 1, y});
                    }
                    if (y > 0 && cond((y - 1) * w + x)) {
                        q.push({x, y - 1});
                    }
                    if (y + 1 < h && cond((y + 1) * w + x)) {
                        q.push({x, y + 1});
                    }
                }
                parts.push_back(part);
            }
        }
    }
    std::sort(parts.begin(), parts.end(), [&](const region_t &a, const region_t &b) {
        int cnt_a = 0;
        int cnt_b = 0;
        for (int i = 0; i < w * h; i++) {
            if (a.r[i]) {
                cnt_a++;
            }
            if (b.r[i]) {
                cnt_b++;
            }
        }
        return cnt_a > cnt_b;
    });
    std::cout << "parts" << parts.size() << std::endl;
#if 0
    for (int i = 0; i < parts.size(); i++) {
        std::vector<unsigned char> out;
        out.resize(src.size());
        const auto &p = parts[i];
        for (int j = 0; j < w * h; j++) {
            if (!p[j]) {
                continue;
            }
            out[j * 4 + 0] = src[j * 4 + 0];
            out[j * 4 + 1] = src[j * 4 + 1];
            out[j * 4 + 2] = src[j * 4 + 2];
            out[j * 4 + 3] = src[j * 4 + 3];
        }
        std::ostringstream oss;
        oss << "debug_" << i << ".png";
        std::string filename = oss.str();
        stbi_write_png(filename.c_str(), w, h, 4, out.data(), w * 4);
    }
    std::cout << "debug" << std::endl;
#endif
    auto cond2 = [&](int index, int base) {
        if (all[index]) {
            return kMaxThreshold + 1;
        }
        int sum = 0;
        sum += diff(src[4 * index + 0], src[4 * base + 0]);
        sum += diff(src[4 * index + 1], src[4 * base + 1]);
        sum += diff(src[4 * index + 2], src[4 * base + 2]);
        sum += diff(src[4 * index + 3], src[4 * base + 3]) * 3 + 1; // alphaは他のより重く見る
        sum += blur[4 * index + 0];
        sum += blur[4 * index + 1];
        sum += blur[4 * index + 2];
        sum += blur[4 * index + 3] * 3;
        return sum;
    };
    auto spread = [&]() {
        std::map<int, std::queue<spread_t>> spread_map;
        for (int j = 0; j < parts.size(); j++) {
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    int i = y * w + x;
                    if (!parts[j].r[i]) {
                        continue;
                    }
                    if (x > 0 && !all[i - 1]) {
                        spread_map[cond2(i - 1, i)].push({x - 1, y, j});
                    }
                    if (x + 1 < w && !all[i + 1]) {
                        spread_map[cond2(i + 1, i)].push({x + 1, y, j});
                    }
                    if (y > 0 && !all[i - w]) {
                        spread_map[cond2(i - w, i)].push({x, y - 1, j});
                    }
                    if (y + 1 < h && !all[i + w]) {
                        spread_map[cond2(i + w, i)].push({x, y + 1, j});
                    }
                }
            }
        }
        int progress = 0;
        int done = 0;
        for (int i = 0; i < w * h; i++) {
            if (all[i]) {
                done++;
            }
        }
        while (done < all.size()) {
            int d = 0;
            std::optional<int> defer_delete;
            for (auto &[k, v] : spread_map) {
                std::queue<spread_t> q;
                while (!v.empty()) {
                    auto [x, y, index] = v.front();
                    v.pop();
                    int i = y * w + x;
                    if (all[i]) {
                        continue;
                    }
                    d++;
                    all[i] = true;
                    parts[index].r[i] = true;
                    if (x > 0 && !all[i - 1]) {
                        auto c = cond2(i - 1, i);
                        if (c == k) {
                            q.push({x - 1, y, index});
                        }
                        else {
                            spread_map[c].push({x - 1, y, index});
                        }
                    }
                    if (x + 1 < w && !all[i + 1]) {
                        auto c = cond2(i + 1, i);
                        if (c == k) {
                            q.push({x + 1, y, index});
                        }
                        else {
                            spread_map[c].push({x + 1, y, index});
                        }
                    }
                    if (y > 0 && !all[i - w]) {
                        auto c = cond2(i - w, i);
                        if (c == k) {
                            q.push({x, y - 1, index});
                        }
                        else {
                            spread_map[c].push({x, y - 1, index});
                        }
                    }
                    if (y + 1 < h && !all[i + w]) {
                        auto c = cond2(i + w, i);
                        if (c == k) {
                            q.push({x, y + 1, index});
                        }
                        else {
                            spread_map[c].push({x, y + 1, index});
                        }
                    }
                }
                v = q;
                if (v.empty()) {
                    defer_delete = k;
                }
                break;
            }
            if (defer_delete) {
                spread_map.erase(defer_delete.value());
                defer_delete = std::nullopt;
            }
            if (progress < (done + d) * 100 / all.size()) {
                progress = (done + d) * 100 / all.size();
                std::cout << progress << "%" << std::endl;
            }
            done += d;
        }
    };
    std::sort(parts.begin(), parts.end(), [&](const region_t &a, const region_t &b) {
        int cnt_a = 0;
        int cnt_b = 0;
        for (int i = 0; i < w * h; i++) {
            if (a.r[i]) {
                cnt_a++;
            }
            if (b.r[i]) {
                cnt_b++;
            }
        }
        return cnt_a > cnt_b;
    });
    spread();
    std::sort(parts.begin(), parts.end(), [&](const region_t &a, const region_t &b) {
        int cnt_a = 0;
        int cnt_b = 0;
        for (int i = 0; i < w * h; i++) {
            if (a.r[i]) {
                cnt_a++;
            }
            if (b.r[i]) {
                cnt_b++;
            }
        }
        return cnt_a > cnt_b;
    });
    int sum = parts.size();
    while (true) {
        auto &last = parts.back();
        int cnt = 0;
        for (int i = 0; i < w * h; i++) {
            if (last.r[i]) {
                cnt++;
            }
        }
        if (cnt < all.size() * rate / 100.0) {
            for (int i = 0; i < w * h; i++) {
                if (last.r[i]) {
                    all[i] = false;
                }
            }
            parts.pop_back();
        }
        else {
            break;
        }
    }
    std::cout << "parts reduction: " << sum << " -> " << parts.size() << std::endl;
    spread();
    std::sort(parts.begin(), parts.end(), [&](const region_t &a, const region_t &b) {
        int cnt_a = 0;
        int cnt_b = 0;
        for (int i = 0; i < w * h; i++) {
            if (a.r[i]) {
                cnt_a++;
            }
            if (b.r[i]) {
                cnt_b++;
            }
        }
        return cnt_a > cnt_b;
    });
    for (int i = 0; i < parts.size(); i++) {
        std::vector<unsigned char> out;
        out.resize(src.size());
        const auto &p = parts[i];
        for (int j = 0; j < w * h; j++) {
            if (!p.r[j]) {
                continue;
            }
            out[j * 4 + 0] = src[j * 4 + 0];
            out[j * 4 + 1] = src[j * 4 + 1];
            out[j * 4 + 2] = src[j * 4 + 2];
            out[j * 4 + 3] = src[j * 4 + 3];
        }
        std::ostringstream oss;
        oss << "output_" << i << ".png";
        std::string filename = oss.str();
        stbi_write_png(filename.c_str(), w, h, 4, out.data(), w * 4);
    }
    std::cout << "save" << std::endl;
    return 0;
}
