#pragma once

#include "mnist.h"

using namespace std;

class Layer {
public:
	Layer(size_t width, size_t height);
	void optimize(MNIST dataset, size_t num_iterations, float learning_rate);
	shared_ptr<float> predict(MNIST dataset);
public:
	float forward(MNIST dataset);
	void backward(size_t m, float learning_rate);

	const size_t width, height;

	shared_ptr<float> d_w;
	shared_ptr<float> d_b;
	shared_ptr<float> d_dw;
	shared_ptr<float> d_db;

	shared_ptr<float> h_w;
	shared_ptr<float> h_b;
	shared_ptr<float> h_dw;
	shared_ptr<float> h_db;
};
