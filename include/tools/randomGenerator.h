#pragma once
#include <random>
#include <vector>
#include <omp.h>

class RandomGenerator {
    static std::vector<std::mt19937_64> generators;

public:
    static void init(unsigned threads) {
        generators.reserve(threads);
        std::random_device rd;
        for (unsigned i = 0; i < threads; ++i) { generators.emplace_back(rd()); }
    }

    [[nodiscard]] static double randomUnit() {
        std::uniform_real_distribution<double> dis(0, 1);
        return dis(generators[omp_get_thread_num()]);
    }

    [[nodiscard]] static double randomReal(double max) {
        std::uniform_real_distribution<double> dis(0, max);
        return dis(generators[omp_get_thread_num()]);
    }

    [[nodiscard]] static unsigned randomUnsigned(unsigned max) {
        std::uniform_int_distribution<unsigned> dis(0, max);
        return dis(generators[omp_get_thread_num()]);
    }
};