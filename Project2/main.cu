#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

#include "mnist.h"
#include "layer.h"
#include "util.h"

#include <iostream>
#include <fstream>

using namespace std;

int main() {
	ifstream train_images_is("E:/study/ECE277/proj4/ppNew Compressed (zipped) Folder/Project2/Data/train-images.idx3-ubyte", std::ios::in | std::ios::binary);
	ifstream train_labels_is("E:/study/ECE277/proj4/ppNew Compressed (zipped) Folder/Project2/Data/train-labels.idx1-ubyte", std::ios::in | std::ios::binary);
	MNIST train_dataset(train_images_is, train_labels_is);
	train_dataset.test(200, 28, 28, false);

	Layer full_connected(28, 28);
	full_connected.optimize(train_dataset, 1000, 1e-3f);

	ifstream test_images_is("E:/study/ECE277/proj4/ppNew Compressed (zipped) Folder/Project2/Data/t10k-images.idx3-ubyte", std::ios::in | std::ios::binary);
	ifstream test_labels_is("E:/study/ECE277/proj4/ppNew Compressed (zipped) Folder/Project2/Data/t10k-labels.idx1-ubyte", std::ios::in | std::ios::binary);
	MNIST test_dataset(test_images_is, test_labels_is);
	test_dataset.test(200, 28, 28, false);

	auto d_y_pred = full_connected.predict(test_dataset);
	auto h_y_pred = allocateOnHost<float>(test_dataset.m);
	cudaMemcpy(h_y_pred.get(), d_y_pred.get(), test_dataset.m * sizeof(float), cudaMemcpyDeviceToHost);

	int correct = 0;
	float threshold = 0.2f;
	for (size_t k = 0; k < test_dataset.m; k++) {
		bool flg = false;
		if ((test_dataset.h_y.get()[k] == 1 && h_y_pred.get()[k] >= 0.5f) || (test_dataset.h_y.get()[k] == 0 && h_y_pred.get()[k] < 0.5f)) {
			++correct;
			flg = true;
		}
		// cout << test_dataset.h_y.get()[k] << " " << h_y_pred.get()[k] << " " << flg << endl;
	}
	float error = 1.0 - ((float)correct / (float)test_dataset.m);
	cout << correct<< " to " <<  test_dataset.m << endl;
	cout << error << endl;
	getchar();

	return 0;
}