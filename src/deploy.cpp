#include "deploy.hpp" 
 
int main(int argc, char** argv){ 
	std::cout << " -- Project: deploy" << std::endl; 
	
	// check parameter
	if (argc < 4){
		std::cout << "Model files are required! Please check and run: " 
		<< std::endl
		<< "\t./deploy [model].param [model].bin" << std::endl;
		return 0;
	}
	else if (argc > 4){
		std::cout << "\tToo much parameters." << std::endl;
		return 0;
	}
	
	// read file names
	const char* paramfile = argv[1];
    const char* binfile = argv[2];
	const char* imagefile = argv[3];

	// build model
	ncnn::Net model;
	model.load_param(paramfile);
	model.load_model(binfile);
	std::cout << "Model loaded." << std::endl;

	// read image
	float mean[3] = {104,117,123};
	float norm[3] = {0.017, 0.017, 0.017};
	cv::Mat cv_image = cv::imread(imagefile);
	ncnn::Mat ncnn_image = ncnn::Mat::from_pixels_resize(cv_image.data, ncnn::Mat::PIXEL_BGR2RGB, 
												  		 cv_image.cols, cv_image.rows, 224, 224);
	ncnn_image.substract_mean_normalize(mean, norm);

	// run inference
	ncnn::Extractor extractor = model.create_extractor();
	ncnn::Mat output;
	extractor.input("input", ncnn_image);
	extractor.extract("output", output);

	std::vector<float> scores;
	for (int i = 0; i < output.w; ++i){
		scores.push_back(output[i]);
	}
	int answer = std::max_element(scores.begin(), scores.end()) - scores.begin();

	std::cout << "result: " << answer << std::endl;
 
	return 0; 
}
