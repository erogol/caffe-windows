#include <cuda_runtime.h>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <string>
#include <iostream>
#include <stdio.h>
#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"
#include "caffe/blob.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc/imgproc.hpp>

#define SENTIO_MODEL_PROTO "C:/Users/eren/Dropbox/SENTIO/cifar_convnet_leveldb_deploy.prototxt"
#define SENTIO_MODEL_BIN "C:/Users/eren/Dropbox/SENTIO/convnet_leveldb_disturb_data_iter_20000.caffemodel"
#define SENTIO_EXP_IMG "C:/Users/eren/Dropbox/SENTIO/player_images/182.jpg"

using namespace caffe;
using namespace std;

int predict() {
	const int img_height = 40;
	const int img_width = 20;

	// Set GPU
	Caffe::set_mode(Caffe::CPU);
	//int device_id = 0;
	//Caffe::SetDevice(device_id);
	LOG(INFO) << "Using GPU";

	// Set to TEST Phase
	Caffe::set_phase(Caffe::TEST);

	// Load net
	Net<float> net(SENTIO_MODEL_PROTO);

	// Load pre-trained net (binary proto)
	net.CopyTrainedLayersFrom(SENTIO_MODEL_BIN);

	// Load image
	cv::Mat image_tmp = cv::imread(SENTIO_EXP_IMG);// or cat.jpg
	cv::Mat image;
	cv::resize(image_tmp, image, cv::Size(img_width, img_height));

	// Set vector for image
	vector<cv::Mat> imageVector;
	imageVector.push_back(image);

	// Set vector for label
	vector<int> labelVector;
	labelVector.push_back(0);//push_back 0 for initialize purpose

	// Net initialization
	float loss = 0.0;
	boost::shared_ptr<MemoryDataLayer<float> > memory_data_layer;
	memory_data_layer = boost::static_pointer_cast<MemoryDataLayer<float> >(net.layer_by_name("data"));

	memory_data_layer->AddMatVector(imageVector, labelVector);

	// Run ForwardPrefilled 
	vector<Blob<float>*> results = net.ForwardPrefilled(&loss);

	// Display result
	const float* argmaxs = results[1]->cpu_data();

	for (int i = 0; i < results[1]->num(); i++)
	{
		for (int j = 0; j < results[1]->height(); j++)
		{
			LOG(INFO) << "Image: " << i << " class:" << argmaxs[i*results[1]->height() + j];
		}
	}

	return 0;
}

int main(int argc, char** argv) {
	predict();
	/*BlobProto blob_proto;
	bool bool_val = ReadProtoFromBinaryFile("C:/Users/eren/Dropbox/SENTIO/sentio_40x20_mean.binaryproto", &blob_proto);
	cout << bool_val << endl;*/
}