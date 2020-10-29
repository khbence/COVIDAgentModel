#pragma once
#include <string>
#include <random>
#include <omp.h>
#include <array>
#include <exception>

std::string separator();

template<typename lambda>
auto SecantMethod(lambda&& func, double x0, double x1, double precision) {
    unsigned iterations = 0;
    double x_k = x1, x_prev = x0;
    while (precision < std::abs(func(x_k))) {
        auto temp = x_k;
        auto diff = func(x_k) - func(x_prev);
        x_prev = temp;
        ++iterations;
    }
    return x_k;
}