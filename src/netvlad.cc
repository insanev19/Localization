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
	float confThreshold; // Confidence threshold
	float nmsThreshold;  // Non-maximum suppression threshold
	float objThreshold;  //Object Confidence threshold
	string modelpath;
};

typedef struct BoxInfo
{
	float x1;
	float y1;
	float x2;
	float y2;
	float score;
	int label;
} BoxInfo;

int endsWith(string s, string sub) {
	return s.rfind(sub) == (s.length() - sub.length()) ? 1 : 0;
}

const float anchors_640[3][6] = { {10.0,  13.0, 16.0,  30.0,  33.0,  23.0},
								 {30.0,  61.0, 62.0,  45.0,  59.0,  119.0},
								 {116.0, 90.0, 156.0, 198.0, 373.0, 326.0} };

const float anchors_1280[4][6] = { {19, 27, 44, 40, 38, 94},{96, 68, 86, 152, 180, 137},{140, 301, 303, 264, 238, 542},
					   {436, 615, 739, 380, 925, 792} };

class YOLO
{
public:
	YOLO(Net_config config);
	void detect(vector<Mat>& frames, int batch_size);
private:
	float* anchors;
	int num_stride;
	int inpWidth;
	int inpHeight;
	int nout;
	int num_proposal;
	vector<string> class_names;
	int num_class;
	int seg_num_class;

	float confThreshold;
	float nmsThreshold;
	float objThreshold;
	const bool keep_ratio = true;
	void normalize_(Mat img, vector<float>& input_image_);
	void nms(vector<BoxInfo>& input_boxes);
	Mat resize_image(Mat srcimg, int* newh, int* neww, int* top, int* left);
   
    Ort::Env env = Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, "test");
    Ort::Session* ort_session = nullptr;
    Ort::SessionOptions sessionOptions = SessionOptions();    
    
	vector<char*> input_names;
	vector<char*> output_names;
	vector<vector<int64_t>> input_node_dims; // >=1 outputs
	vector<vector<int64_t>> output_node_dims; // >=1 outputs
};

YOLO::YOLO(Net_config config)
{
	this->confThreshold = config.confThreshold;
	this->nmsThreshold = config.nmsThreshold;
	this->objThreshold = config.objThreshold;
    
	string classesFile = "coco.names";
	string model_path = "yolov5s.onnx";
    
    OrtCUDAProviderOptions cuda_options{};
    sessionOptions.AppendExecutionProvider_CUDA(cuda_options);

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

	this->inpHeight = input_node_dims[0][2];
	this->inpWidth = input_node_dims[0][3];
	this->nout = output_node_dims[0][2];
	this->num_proposal = output_node_dims[0][1];

	ifstream ifs(classesFile.c_str());
	string line;
	while (getline(ifs, line)) this->class_names.push_back(line);
	this->num_class = class_names.size();

	if (endsWith(config.modelpath, "6.onnx"))
	{
		anchors = (float*)anchors_1280;
		this->num_stride = 4;
	}
	else
	{
		anchors = (float*)anchors_640;
		this->num_stride = 3;
	}
}

Mat YOLO::resize_image(Mat srcimg, int* newh, int* neww, int* top, int* left)
{
	int srch = srcimg.rows, srcw = srcimg.cols;
	*newh = this->inpHeight;
	*neww = this->inpWidth;
	Mat dstimg;
	if (this->keep_ratio && srch != srcw) {
		float hw_scale = (float)srch / srcw;
		if (hw_scale > 1) {
			*newh = this->inpHeight;
			*neww = int(this->inpWidth / hw_scale);
			resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
			*left = int((this->inpWidth - *neww) * 0.5);
			copyMakeBorder(dstimg, dstimg, 0, 0, *left, this->inpWidth - *neww - *left, BORDER_CONSTANT, 114);
		}
		else {
			*newh = (int)this->inpHeight * hw_scale;
			*neww = this->inpWidth;
			resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
			*top = (int)(this->inpHeight - *newh) * 0.5;
			copyMakeBorder(dstimg, dstimg, *top, this->inpHeight - *newh - *top, 0, 0, BORDER_CONSTANT, 114);
		}
	}
	else {
		resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
	}
	return dstimg;
}

void YOLO::normalize_(Mat img, vector<float>& input_image_)
{
	//    img.convertTo(img, CV_32F);
	int row = img.rows;
	int col = img.cols;
	input_image_.resize(row * col * img.channels());
	for (int c = 0; c < 3; c++)
	{
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++)
			{
				float pix = img.ptr<uchar>(i)[j * 3 + 2 - c];
				input_image_[c * row * col + i * col + j] = pix / 255.0;

			}
		}
	}
}

void YOLO::nms(vector<BoxInfo>& input_boxes)
{
	sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b) { return a.score > b.score; });
	vector<float> vArea(input_boxes.size());
	for (int i = 0; i < int(input_boxes.size()); ++i)
	{
		vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
			* (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
	}

	vector<bool> isSuppressed(input_boxes.size(), false);
	for (int i = 0; i < int(input_boxes.size()); ++i)
	{
		if (isSuppressed[i]) { continue; }
		for (int j = i + 1; j < int(input_boxes.size()); ++j)
		{
			if (isSuppressed[j]) { continue; }
			float xx1 = (max)(input_boxes[i].x1, input_boxes[j].x1);
			float yy1 = (max)(input_boxes[i].y1, input_boxes[j].y1);
			float xx2 = (min)(input_boxes[i].x2, input_boxes[j].x2);
			float yy2 = (min)(input_boxes[i].y2, input_boxes[j].y2);

			float w = (max)(float(0), xx2 - xx1 + 1);
			float h = (max)(float(0), yy2 - yy1 + 1);
			float inter = w * h;
			float ovr = inter / (vArea[i] + vArea[j] - inter);

			if (ovr >= this->nmsThreshold)
			{
				isSuppressed[j] = true;
			}
		}
	}
	// return post_nms;
	int idx_t = 0;
	input_boxes.erase(remove_if(input_boxes.begin(), input_boxes.end(), [&idx_t, &isSuppressed](const BoxInfo& f) { return isSuppressed[idx_t++]; }), input_boxes.end());
}

void YOLO::detect(vector<Mat>& frames, int batch_size)
{
	vector<int>newhs, newws, padhs, padws;

	auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	array<int64_t, 4> input_shape_{ 1, 3, this->inpHeight, this->inpWidth };
	std::vector<Ort::Value> inputBatchTensors;

	for (int b_idx = 0; b_idx < batch_size; b_idx++) {
		vector<float> input_image_;
		int newh = 0, neww = 0, padh = 0, padw = 0;
		Mat dstimg = this->resize_image(frames[b_idx], &newh, &neww, &padh, &padw);
		this->normalize_(dstimg, input_image_);
		newhs.push_back(newh);
		newws.push_back(neww);
		padhs.push_back(padh);
		padws.push_back(padw);

		inputBatchTensors.push_back(Value::CreateTensor<float>(allocator_info, input_image_.data(),
			input_image_.size(), input_shape_.data(), input_shape_.size()));
	}

	std::vector<const char*> inputBatchNames, outputBatchNames;

	for (size_t i = 0; i < batch_size; i++) {
		inputBatchNames.insert(inputBatchNames.end(), input_names.begin(), input_names.end());
		outputBatchNames.insert(outputBatchNames.end(), output_names.begin(), output_names.end());
	}
		// 开始推理
	vector<Value> ort_outputs = ort_session->Run(RunOptions{ nullptr }, inputBatchNames.data(), inputBatchTensors.data(), batch_size, 
																				outputBatchNames.data(), outputBatchNames.size());   // 开始推理

   	for(int pos_idx = 0; pos_idx < batch_size; pos_idx++){
		//generate proposals
		vector<BoxInfo> generate_boxes;
		float ratioh = (float)frames[pos_idx].rows / newhs[pos_idx], ratiow = (float)frames[pos_idx].cols / newws[pos_idx];
		int n = 0, q = 0, i = 0, j = 0, row_ind = 0, k = 0; ///xmin,ymin,xamx,ymax,box_score, class_score
		const float* pdata = ort_outputs[pos_idx].GetTensorMutableData<float>();
		for (n = 0; n < this->num_stride; n++)   ///特征图尺度
		{
			const float stride = pow(2, n + 3);
			int num_grid_x = (int)ceil((this->inpWidth / stride));
			int num_grid_y = (int)ceil((this->inpHeight / stride));
			for (q = 0; q < 3; q++)    ///anchor
			{
				const float anchor_w = this->anchors[n * 6 + q * 2];
				const float anchor_h = this->anchors[n * 6 + q * 2 + 1];
				for (i = 0; i < num_grid_y; i++)
				{
					for (j = 0; j < num_grid_x; j++)
					{
						float box_score = pdata[4];
                    
						if (box_score > this->objThreshold)
						{
							int max_ind = 0;
							float max_class_socre = 0;
							for (k = 0; k < num_class; k++)
							{
								if (pdata[k + 5] > max_class_socre)
								{
									max_class_socre = pdata[k + 5];
									max_ind = k;
								}
							}
							max_class_socre *= box_score;
							if (max_class_socre > this->confThreshold)
							{
								float cx = (pdata[0] * 2.f - 0.5f + j) * stride;  ///cx
								float cy = (pdata[1] * 2.f - 0.5f + i) * stride;   ///cy
								float w = powf(pdata[2] * 2.f, 2.f) * anchor_w;   ///w
								float h = powf(pdata[3] * 2.f, 2.f) * anchor_h;  ///h

								float xmin = (cx - padws[pos_idx] - 0.5 * w) * ratiow;
								float ymin = (cy - padhs[pos_idx] - 0.5 * h) * ratioh;
								float xmax = (cx - padws[pos_idx] + 0.5 * w) * ratiow;
								float ymax = (cy - padhs[pos_idx] + 0.5 * h) * ratioh;

								generate_boxes.push_back(BoxInfo{ xmin, ymin, xmax, ymax, max_class_socre, max_ind });
							}
						}
						row_ind++;
						pdata += nout;
					}
				}
			}
		}

		// Perform non maximum suppression to eliminate redundant overlapping boxes with
		// lower confidences
		nms(generate_boxes);

		for (size_t i = 0; i < generate_boxes.size(); ++i)
		{
			int xmin = int(generate_boxes[i].x1);
			int ymin = int(generate_boxes[i].y1);

			 rectangle(frames[pos_idx], Point(xmin, ymin), Point(int(generate_boxes[i].x2), int(generate_boxes[i].y2)), Scalar(0, 0, 255), 2);
			 string label = format("%.2f", generate_boxes[i].score);
			 label = this->class_names[generate_boxes[i].label] + ":" + label;
			 putText(frames[pos_idx], label, Point(xmin, ymin - 5), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255, 0), 1);
		}
		
	}
}

int main()
{
	Net_config yolo_nets = { 0.3, 0.5, 0.3,"./yolov5s.onnx" };
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

