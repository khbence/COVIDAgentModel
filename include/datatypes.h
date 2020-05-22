#pragma once
#ifndef THRUST_DEVICE_SYSTEM
#define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_CUDA
#endif

#define HD __device__ __host__

#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_OMP
//#define HD
#elif THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
#define HD __host__ __device__
#endif
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/count.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/binary_search.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>