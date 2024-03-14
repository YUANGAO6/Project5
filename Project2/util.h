#pragma once

#include <memory>
#include <istream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

using namespace std;

template<typename T>
shared_ptr<T> allocateOnDevice(size_t size) {
	T* p;
	cudaMalloc(&p, size * sizeof(T));
	return shared_ptr<T>(p, [](T* ptr) { cudaFree(ptr); });
}

template<typename T>
shared_ptr<T> allocateOnHost(size_t size) {
	T* p = new T[size];
	return shared_ptr<T>(p, [](T* ptr) { delete[] ptr; });
}

template<typename T>
T sumOnHost(T* data, size_t size) {
	auto s_p = allocateOnHost<T>(size);
	cudaMemcpy(s_p.get(), data, size * sizeof(T), cudaMemcpyDeviceToHost);
	T sum = T();
	for (size_t i = 0; i < size; ++i) {
		sum += s_p.get()[i];
	}
	return sum;
}


template<typename T>
void memsetOnHost(T* ts, T val, size_t size) {
	for (size_t i = 0; i < size; i++) {
		ts[i] = val;
	}
}

istream& read_int32(istream& is, int32_t& x);
