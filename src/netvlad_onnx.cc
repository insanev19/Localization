#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
//#include <cuda_provider_factory.h>
#include <onnxruntime_cxx_api.h>
#include <assert.h>
#include <chrono>
#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>

#include <string>
#include <vector>
#include <iomanip>
#include <thread>
#include <ctime>


using namespace std;
using namespace cv;
using namespace Ort;

struct Net_config
{
	int max_size;
	string modelpath;
};

int endsWith(string s, string sub) {
	return s.rfind(sub) == (s.length() - sub.length()) ? 1 : 0;
}

class NetVLAD
{
public:
    NetVLAD(Net_config);
    vector<float> extract(Mat img);

private:

    int max_size;

    Mat resize_image(Mat src_img, int max_size);

    Ort::Env env = Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, "test");
    Ort::Session* ort_session = nullptr;
    Ort::SessionOptions sessionOptions = SessionOptions();  

    vector<char*> input_names;
	vector<char*> output_names;
	vector<vector<int64_t>> input_node_dims; // >=1 outputs
	vector<vector<int64_t>> output_node_dims; // >=1 outputs  
};

NetVLAD::NetVLAD(Net_config config){
    this->max_size = config.max_size;
    string model_path = config.modelpath;

    ort_session = new Session(env, model_path.c_str(), sessionOptions);

	size_t numInputNodes = ort_session->GetInputCount();
	size_t numOutputNodes = ort_session->GetOutputCount();
	AllocatorWithDefaultOptions allocator;
	for (int i = 0; i < numInputNodes; i++)
	{
		input_names.push_back(ort_session->GetInputName(i, allocator));
		Ort::TypeInfo input_type_info = ort_session->GetInputTypeInfo(i);
		auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
		auto input_dims = input_tensor_info.GetShape();
		input_node_dims.push_back(input_dims);
		
	}

	for (int i = 0; i < numOutputNodes; i++)
	{
		output_names.push_back(ort_session->GetOutputName(i, allocator));
		Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
		auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
		auto output_dims = output_tensor_info.GetShape();
		output_node_dims.push_back(output_dims);
	}

}

int main()
{
	Net_config yolo_nets = { 640 ,"./netvlad_dynamic_input.onnx" };
	YOLO yolo_model(yolo_nets);
	string imgpath = "./bus.jpg";
	Mat srcimg = imread(imgpath);
	int batchsize = 2048;
	vector<Mat> frames;
	for (int k = 0; k < batchsize; k++) {
		frames.push_back(srcimg);
	}

	clock_t start = clock();
	for (int i = 0; i < 10; i++) {
		yolo_model.detect(frames, batchsize);
		std::cout << " --------all step is 10000  now is--- " << i << std::endl;
	}
	clock_t end = clock();
	std::cout << "10000 inference  takes " << (double)(end - start) /( CLOCKS_PER_SEC) << " s" << std::endl;
    cv::imwrite("result.jpg",srcimg);
    
    return 0;
}

