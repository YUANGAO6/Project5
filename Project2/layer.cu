#include "layer.h"
#include "util.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

#include <iostream>
#include <cassert>

using namespace std;

const size_t x_threads = 8;
const size_t y_threads = 8;
const size_t z_threads = 1;
const size_t threads_num = x_threads * y_threads * z_threads;

Layer::Layer(size_t width, size_t height) :
	width(width), height(height),
	h_w(allocateOnHost<float>(width* height)),
	h_b(allocateOnHost<float>(1)),
	h_dw(allocateOnHost<float>(width* height)),
	h_db(allocateOnHost<float>(1)),
	d_w(allocateOnDevice<float>(width* height)),
	d_b(allocateOnDevice<float>(1)),
	d_dw(allocateOnDevice<float>(width* height)),
	d_db(allocateOnDevice<float>(1)) {

	memsetOnHost(h_w.get(), 0.0f, width * height);
	memsetOnHost(h_b.get(), 0.0f, 1);
	memsetOnHost(h_dw.get(), 0.0f, width * height);
	memsetOnHost(h_db.get(), 0.0f, 1);

	cudaMemcpy(d_w.get(), h_w.get(), width * height * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b.get(), h_b.get(), sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_dw.get(), h_dw.get(), width * height * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_db.get(), h_db.get(), sizeof(float), cudaMemcpyHostToDevice);
}

__device__ __host__ float sigmoid(float x) {
	return 1.0f / (1 + expf(-x));
}

// blocks(1, 1, m)
// threads(NUM_THREADS_X, NUM_THREADS_X, 1)
// 
__global__ void product(float* x, float* w, float* b, float* y, size_t width, size_t height, size_t m) {
	__shared__ float cache[threads_num];

	size_t i = threadIdx.x;
	size_t j = threadIdx.y;
	size_t k = blockIdx.z;

	size_t cid = (j * x_threads) + i;

	float temp = 0.0f;
	for (auto ii = i; ii < width; ii += blockDim.x) {
		for (auto jj = j; jj < height; jj += blockDim.y) {
			size_t wid = (jj * width) + ii;
			size_t xid = (k * width * height) + wid;
			temp += w[wid] * x[xid];
		}
	}
	cache[cid] = temp;
	__syncthreads();

	size_t size = threads_num;
	for (size_t half = size / 2; half > 0; half /= 2) {
		if (cid < half) {
			cache[cid] += cache[cid + half];
		}
		__syncthreads();
	}

	if (cid == 0) {
		float a = sigmoid(cache[0] + *b);
		assert(a != 0.0f && a != 1.0f);
		y[k] = a;
	}
}


// blocks(m/NUM_THREADS, 1, 1)
// threads(NUM_THREADS, 1, 1)
__global__ void cost(float* y, float* a, float* cost_partial, size_t m) {
	__shared__ float cache[threads_num];

	size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
	size_t cid = threadIdx.x;
	size_t bid = blockIdx.x;

	float temp = 0.0f;
	if (tid < m) {
		temp = y[tid] * logf(a[tid]) + (1 - y[tid]) * (logf(1 - a[tid]));
	}
	cache[cid] = temp;
	__syncthreads();

	size_t size = threads_num;
	for (size_t half = size / 2; half > 0; half /= 2) {
		if (cid < half) {
			cache[cid] += cache[cid + half];
		}
		__syncthreads();
	}

	if (cid == 0) {
		cost_partial[bid] = -cache[0];
	}
}


// blocks(size/NUM_THREADS)
// threads(NUM_THREADS)
__global__ void memsetOnDevice(float* ts, float val, size_t size) {
	size_t i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < size) {
		ts[i] = val;
	}
}


// blocks(width/NUM_THREADS_X, height/NUM_THREADS_Y, m)
// threads(NUM_THREADS_X, NUM_THREADS_Y, 1)
__global__ void weights_gradient(float* x, float* y, float* a, float* dw, size_t width, size_t height) {
	size_t i = blockDim.x * blockIdx.x + threadIdx.x;
	size_t j = blockDim.y * blockIdx.y + threadIdx.y;
	size_t k = blockIdx.z;

	size_t wid = (j * width) + i;
	size_t xid = (k * width * height) + wid;


	atomicAdd(&dw[wid], x[xid] * (a[k] - y[k]));
}


// blocks(m/NUM_THREADS, 1, 1)
// threads(NUM_THREADS, 1, 1)
__global__ void bias_gradient(float* y, float* a, float* db_partial, size_t m) {
	__shared__ float cache[threads_num];

	size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
	size_t cid = threadIdx.x;
	size_t bid = blockIdx.x;

	float temp = 0.0f;
	if (tid < m) {
		temp = a[tid] - y[tid];
	}
	cache[cid] = temp;
	__syncthreads();

	size_t size = threads_num;
	for (size_t half = size / 2; half > 0; half /= 2) {
		if (cid < half) {
			cache[cid] += cache[cid + half];
		}
		__syncthreads();
	}

	if (cid == 0) {
		db_partial[bid] = cache[0];
	}
}


void Layer::forward(MNIST dataset) {
	size_t width = dataset.width;
	size_t height = dataset.height;
	size_t m = dataset.m;

	auto& d_x = dataset.d_x;
	auto& d_y = dataset.d_y;
	auto& h_x = dataset.h_x;
	auto& h_y = dataset.h_y;

	dim3 threads(x_threads, y_threads, 1);
	dim3 blocks(1, 1, m);
	auto d_a = allocateOnDevice<float>(m);
	product << <blocks, threads >> > (d_x.get(), d_w.get(), d_b.get(), d_a.get(), width, height, m);
	cudaDeviceSynchronize();



	threads = threads_num;
	blocks = (m + threads.x - 1) / threads.x;
	auto d_cost_partial = allocateOnDevice<float>(blocks.x);
	cost << <blocks, threads >> > (d_y.get(), d_a.get(), d_cost_partial.get(), m);
	cudaDeviceSynchronize();

	auto h_cost = sumOnHost<float>(d_cost_partial.get(), blocks.x) / m;

	cout << "Cost: " << h_cost << endl;

	threads = threads_num;
	blocks = (width * height + threads.x - 1) / threads.x;
	memsetOnDevice << <blocks, threads >> > (d_dw.get(), 0.0f, width * height);
	cudaDeviceSynchronize();

	threads = { (unsigned int)x_threads, (unsigned int)y_threads, 1 };
	blocks = {
		(unsigned int)((width + threads.x - 1) / threads.x),
		(unsigned int)((height + threads.y - 1) / threads.y),
		(unsigned int)m
	};
	weights_gradient << <blocks, threads >> > (d_x.get(), d_y.get(), d_a.get(), d_dw.get(), width, height);
	cudaDeviceSynchronize();


	threads = threads_num;
	blocks = (m + threads.x - 1) / threads.x;
	auto d_db_partial = allocateOnDevice<float>(blocks.x);
	bias_gradient << <blocks, threads >> > (d_y.get(), d_a.get(), d_db_partial.get(), m);
	cudaDeviceSynchronize();
	auto h_db = sumOnHost<float>(d_db_partial.get(), blocks.x) / m;
	cudaMemcpy(d_db.get(), &h_db, sizeof(float), cudaMemcpyHostToDevice);


}

// blocks(width/NUM_THREADS_X, height/NUM_THREADS_Y, m)
// threads(NUM_THREADS_X, NUM_THREADS_Y, 1)
__global__ void weights_backward(float* w, float* dw, size_t width, size_t height, size_t m, float learning_rate) {
	size_t i = blockDim.x * blockIdx.x + threadIdx.x;
	size_t j = blockDim.y * blockIdx.y + threadIdx.y;
	size_t wid = (j * width) + i;

	if (i < width && j < height) {
		w[wid] -= learning_rate * dw[wid] / m;
	}
}

void weights_backwardOnHost(float* w, float* dw, size_t width, size_t height, size_t m, float learning_rate) {
	for (size_t j = 0; j < height; j++) {
		for (size_t i = 0; i < width; i++) {
			size_t wid = (j * width) + i;
			w[wid] -= learning_rate * dw[wid] / m;
		}
	}
}

// blocks(1)
// threads(1)
__global__ void bias_backward(float* b, float* db, float learning_rate) {
	b[0] -= learning_rate * db[0];
}

void Layer::backward(size_t m, float learning_rate) {
	dim3 threads(x_threads, y_threads, 1);
	dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y, 1);
	weights_backward << <blocks, threads >> > (d_w.get(), d_dw.get(), width, height, m, learning_rate);

	weights_backwardOnHost(h_w.get(), h_dw.get(), width, height, m, learning_rate);

	threads = 1;
	blocks = 1;
	bias_backward << <blocks, threads >> > (d_b.get(), d_db.get(), learning_rate);
}

void Layer::optimize(MNIST dataset, size_t num_iterations, float learning_rate) {
	for (size_t i = 0; i < num_iterations; ++i) {
		forward(dataset);
		backward(dataset.m, learning_rate);
	}
}

std::shared_ptr<float> Layer::predict(MNIST dataset) {
	std::shared_ptr<float> d_y_pred = allocateOnDevice<float>(dataset.m);

	dim3 threads(x_threads, y_threads, 1);
	dim3 blocks(1, 1, dataset.m);
	product << <blocks, threads >> > (dataset.d_x.get(), d_w.get(), d_b.get(), d_y_pred.get(), dataset.width, dataset.height, dataset.m);
	cudaDeviceSynchronize();

	return d_y_pred;
}