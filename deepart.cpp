// C/C++ File
// AUTHOR:   yuewu
// FILE:     deepart.cu
// CREATED:  2015-09-29 22:48:42
// MODIFIED: 2015-09-29 22:48:42

#include "deepart/deepart.h"
#include "deepart/math_kernels.h"
#include "cpu_lbfgs_helper.h"
#include "error_code.h"

#include "caffe/util/io.hpp"
#include "caffe/util/upgrade_proto.hpp"

#include <CudaLBFGS/lbfgs.h>

#include <algorithm>

using namespace caffe;
using cv::Mat;

namespace deepart {

DeepArt::DeepArt(const std::shared_ptr<caffe::Net<float>>& caffe_net)
	: _caffe_net(caffe_net) {
	if (caffe_net != nullptr) {
		const vector<Blob<float>*>& net_input_blobs =
			this->_caffe_net->input_blobs();
		this->_input_height = net_input_blobs[0]->shape(2);
		this->_input_width = net_input_blobs[0]->shape(3);
	} else {
		this->_input_height = 0;
		this->_input_width = 0;
	}
}

// DeepArt::~DeepArt() {
// 	for (Blob<float>* blob : _style_feat_covariances) {
// 		delete blob;
// 	}
}

std::shared_ptr<caffe::Net<float>> DeepArt::CreateCaffeNet(
	int gpu_id, const string& model_def_path, const string& weight_path) {
	Caffe::SetDevice(gpu_id);
	Caffe::set_mode(Caffe::GPU);

	NetParameter param;
	ReadNetParamsFromTextFileOrDie(model_def_path, &param);
	param.mutable_state()->set_phase(caffe::TEST);
	param.set_force_backward(true);

	std::shared_ptr<caffe::Net<float>> caffe_net(new Net<float>(param));
	caffe_net->CopyTrainedLayersFrom(weight_path);

	return caffe_net;
}

int DeepArt::Init(const string& style_param_file,
				  const string& style_image_path) {
	if (this->_caffe_net == nullptr) {
		return Status_UnInitialized;
	}

	int status_code = Status_OK;
	if (ReadProtoFromTextFile(style_param_file.c_str(),
							  &(this->_style_param)) == false) {
		LOG(ERROR) << "Failed to parse StyleParameter file: "
				   << style_param_file;
		status_code = Status_IOError;
	}
	if (status_code != Status_OK) return status_code;

	// 1. check layer names
	int layer_idx = 0;
	map<string, int> layer_names_index_;
	for (const string& layer_name : this->_caffe_net->layer_names()) {
		layer_names_index_[layer_name] = layer_idx++;
	}

	for (const deepart::LayerParameter& layer :
		 this->_style_param.style_layers()) {
		if (this->_caffe_net->has_layer(layer.name()) == false) {
			LOG(ERROR) << "Unknown style layer name " << layer.name();
			status_code = Status_Invalid_Argument;
		} else {
			this->_style_layer_ids.push_back(layer_names_index_[layer.name()]);
			this->_style_layer_weights.push_back(layer.weight());
		}
	}

	for (const deepart::LayerParameter& layer :
		 this->_style_param.content_layers()) {
		if (this->_caffe_net->has_layer(layer.name()) == false) {
			LOG(ERROR) << "Unknown content layer name " << layer.name();
			status_code = Status_Invalid_Argument;
		} else {
			this->_content_layer_ids.push_back(
				layer_names_index_[layer.name()]);
			this->_content_layer_weights.push_back(layer.weight());
		}
	}
	if (status_code != Status_OK) return status_code;

	// 4. set id of max layer that should be forwarded to
	this->_max_layer_id = 0;
	if (this->_style_layer_ids.size() > 0) {
		this->_max_layer_id =
			std::max(this->_max_layer_id,
					 *std::max_element(this->_style_layer_ids.begin(),
									   this->_style_layer_ids.end()));
	}
	if (this->_content_layer_ids.size() > 0) {
		this->_max_layer_id =
			std::max(this->_max_layer_id,
					 *std::max_element(this->_content_layer_ids.begin(),
									   this->_content_layer_ids.end()));
	}
	if (this->_max_layer_id < this->_caffe_net->layers().size() - 1) {
		++this->_max_layer_id;
	}

	// 5. init style
	status_code = this->InitStyle(style_image_path);

	return status_code;
}

// int DeepArt::InitStyle(const string& style_img_path) {
// 	cv::Mat img = cv::imread(style_img_path, CV_LOAD_IMAGE_COLOR);
// 	if (img.data == nullptr) {
// 		LOG(ERROR) << "unable to load image from " << style_img_path;
// 		return Status_IOError;
// 	}
// 	Blob<float> style_image;
// 	int status_code = this->ImageToBlob(style_image, img);
// 	if (status_code != Status_OK) return status_code;

// 	// 2. load style image and extract covariance feature
// 	vector<Blob<float>*> style_feature_maps;
// 	this->ExtractFeatureMaps(style_feature_maps, this->_style_layer_ids,
// 							 style_image, false);

// 	this->_style_feat_covariances.clear();
// 	for (Blob<float>* feature_map : style_feature_maps) {
// 		Blob<float>* cov_mat = new Blob<float>;
// 		this->ComputeCovarianceMatrix(cov_mat, feature_map);
// 		this->_style_feat_covariances.push_back(cov_mat);
// 	}
// 	return status_code;
// }

int DeepArt::Draw(const string& output_path, const string& img_path,
				  const string& opti_method, int iter_num) {
	LOG(INFO) << "Draw for image(" << img_path
			  << "), output path: " << output_path;

	cv::Mat img = cv::imread(img_path, CV_LOAD_IMAGE_COLOR);
	if (img.data == nullptr) {
		LOG(ERROR) << "unable to load image from " << img_path;
		return Status_IOError;
	}
	int status_code = this->Draw(img, opti_method, iter_num);
	if (status_code == Status_OK) {
		imwrite(output_path, img);
	}
	return status_code;
}

// int DeepArt::Draw(cv::Mat& img, const string& opti_method, int iter_num) {
// 	if (this->_caffe_net == nullptr) {
// 		LOG(ERROR)<<"caffe net not initialized";
// 		return Status_UnInitialized;
// 	}

// 	int status_code = 0;
// 	// 1. load content image and extract content feature maps
// 	Blob<float> content_image;
// 	status_code = this->ImageToBlob(content_image, img);
// 	if (status_code != Status_OK) return status_code;

// 	vector<Blob<float>*> content_feature_maps;
// 	this->ExtractFeatureMaps(content_feature_maps, this->_content_layer_ids,
// 							 content_image);

// 	// 2. Generate input_blob (output image)
// 	Blob<float>* input_blob = this->_caffe_net->input_blobs()[0];
// 	this->GenerateOutputImage(*input_blob, content_image);

// 	// 3. iteration
// 	ArtLossGrad loss_function(this, input_blob->count(1, 4),
// 							  content_feature_maps);
// 	if (opti_method == "sgd") {
// 		float abs_diff;
// 		Blob<float> loss(1, 1, 1, 1);
// 		for (int i = 0; i < iter_num; ++i) {
// 			loss_function.f_gradf(input_blob->gpu_data(),
// 								  loss.mutable_gpu_data(),
// 								  input_blob->mutable_gpu_diff());

// 			caffe_gpu_axpy(input_blob->count(),
// 						   float(-1 * this->_style_param.learning_rate()),
// 						   input_blob->gpu_diff(),
// 						   input_blob->mutable_gpu_data());

// 			this->NormalizePixelRange(input_blob->mutable_gpu_data(),
// 									  input_blob->shape(1),
// 									  input_blob->count(2, 4));
// 		}
// 	} else if (opti_method == "lbfgs") {
// 		float gradientEps = 1e-3f;
// 		cudalbfgs::lbfgs minimizer(loss_function);
// 		minimizer.setMaxIterations(iter_num);
// 		minimizer.setGradientEpsilon(gradientEps);

// 		cudalbfgs::lbfgs::status stat =
// 			minimizer.minimize(input_blob->mutable_gpu_data());

// 		LOG(INFO) << cudalbfgs::lbfgs::statusToString(stat);

// 		if (stat == cudalbfgs::lbfgs::LBFGS_LINE_SEARCH_FAILED) {
// 			status_code = Status_Error;
// 		} else {
// 			this->NormalizePixelRange(input_blob->mutable_gpu_data(),
// 									  input_blob->shape(1),
// 									  input_blob->count(2, 4));
// 		}
// 	}
// #ifdef USE_CPU_LBFGS
// 	else if (opti_method == "c-lbfgs") {
// 		int i, ret = 0;
// 		lbfgsfloatval_t fx;
// 		lbfgs_parameter_t param;

// 		/* Initialize the parameters for the L-BFGS optimization. */
// 		lbfgs_parameter_init(&param);
// 		param.max_iterations = iter_num;
// 		param.m = 3;
// 		param.epsilon = 1e-3;
// 		/*param.linesearch = LBFGS_LINESEARCH_BACKTRACKING;*/

// 		/*
// 		   Start the L-BFGS optimization; this will invoke the callback
// 		   functions
// 		   evaluate() and progress() when necessary.
// 		   */
// 		ret = lbfgs(input_blob->count(), input_blob->mutable_cpu_data(), &fx,
// 					evaluate, progress, &loss_function, &param);
// 		/* Report the result. */
// 		printf("L-BFGS optimization terminated with status code = %d\n", ret);
// 		printf("  fx = %f, x[0] = %f, x[1] = %f\n", fx,
// 			   input_blob->cpu_data()[0], input_blob->cpu_data()[1]);
// 		if (ret != 0) {
// 			status_code = Status_Error;
// 		} else {
// 			this->NormalizePixelRange(input_blob->mutable_gpu_data(),
// 									  input_blob->shape(1),
// 									  input_blob->count(2, 4));
// 		}
// 	}
// #endif
// 	else {
// 		LOG(ERROR) << "unrecognized optimization method " << opti_method;
// 		status_code = Status_Invalid_Argument;
// 	}
// 	if (status_code != Status_OK) return status_code;

// 	// 4. output image
// 	this->BlobToImage(img, *input_blob);

// 	for (Blob<float>* feature_map : content_feature_maps) {
// 		delete feature_map;
// 	}
// 	return status_code;
// }

// int DeepArt::ImageToBlob(Blob<float>& dst_img, const cv::Mat& src_img) {
// 	int channels = src_img.channels();
// 	if (channels != 3) {
// 		LOG(ERROR)
// 			<< "only three channel colorful images are supported, input is "
// 			<< channels;
// 		return Status_Invalid_Input;
// 	}

// 	cv::Mat tmp_img;
// 	cv::resize(src_img, tmp_img,
// 			   cv::Size(this->_input_width, this->_input_height));

// 	dst_img.Reshape(vector<int>{1, 3, this->_input_height, this->_input_width});

// 	int pixel_num = dst_img.count(2, 4);
// 	for (int c = 0; c < channels; ++c) {
// 		const unsigned char* src_ptr = tmp_img.data + c;
// 		float* dst_ptr = dst_img.mutable_cpu_data() + dst_img.offset(0, c);
// 		float mean_val = this->_style_param.mean_values(c);

// 		for (int i = 0; i < pixel_num; ++i, ++dst_ptr, src_ptr += channels) {
// 			*dst_ptr = *src_ptr - mean_val;
// 		}
// 	}
// 	return Status_OK;
// }

// int DeepArt::BlobToImage(cv::Mat& dst_img, Blob<float>& src_blob) {
// 	int ori_height = dst_img.rows;
// 	int ori_width = dst_img.cols;

// 	int channels = src_blob.shape(1);
// 	int pixel_num = src_blob.count(2, 4);
// 	dst_img.create(src_blob.shape(2), src_blob.shape(3), CV_8UC3);

// 	for (int c = 0; c < channels; ++c) {
// 		float mean_value = this->_style_param.mean_values(c);
// 		const float* src_ptr = src_blob.cpu_data() + src_blob.offset(0, c);
// 		unsigned char* dst_ptr = dst_img.data + c;
// 		for (int i = 0; i < pixel_num; ++i, ++src_ptr, dst_ptr += channels) {
// 			*dst_ptr = static_cast<unsigned char>(*src_ptr + mean_value);
// 		}
// 	}

// 	cv::resize(dst_img, dst_img, cv::Size(ori_width, ori_height));

// 	return Status_OK;
// }

// void DeepArt::NormalizePixelRange(float* pixels, int channels, int pixel_num) {
// 	for (int k = 0; k < channels; ++k) {
// 		float min_val = -this->_style_param.mean_values(k);
// 		LimitImagePixelRange(pixel_num, pixels, min_val, min_val + 255.f);
// 		pixels += pixel_num;
// 	}
// }

// void DeepArt::ExtractFeatureMaps(vector<Blob<float>*>& feature_maps,
// 								 const vector<int>& layer_ids,
// 								 const Blob<float>& image, bool is_copy) {

// 	feature_maps.clear();

// 	const vector<Blob<float>*>& net_input_blobs =
// 		this->_caffe_net->input_blobs();
// 	net_input_blobs[0]->CopyFrom(image);

// 	this->_caffe_net->ForwardFromTo(0, this->_max_layer_id);

// 	for (int layer_id : layer_ids) {
// 		if (is_copy == true) {
// 			Blob<float>* feature_map = new Blob<float>;
// 			feature_map->CopyFrom(*(this->_caffe_net->top_vecs()[layer_id][0]),
// 								  false, true);
// 			feature_maps.push_back(feature_map);
// 		} else {
// 			feature_maps.push_back(this->_caffe_net->top_vecs()[layer_id][0]);
// 		}
// 	}
// }

// void DeepArt::ComputeCovarianceMatrix(Blob<float>* covariance_mat,
// 									  const Blob<float>* feature_map) {
// 	covariance_mat->Reshape(1, 1, feature_map->shape(1), feature_map->shape(1));
// 	caffe_gpu_gemm<float>(CblasNoTrans, CblasTrans, feature_map->shape(1),
// 						  feature_map->shape(1),
// 						  feature_map->shape(2) * feature_map->shape(3), 1,
// 						  feature_map->gpu_data(), feature_map->gpu_data(), 0,
// 						  covariance_mat->mutable_gpu_data());
// }

// void DeepArt::GenerateOutputImage(Blob<float>& dst_img,
// 								  const Blob<float>& content_img) {
// 	dst_img.Reshape(content_img.shape());
// 	dst_img.CopyFrom(content_img);
// 	// memset(dst_img.mutable_cpu_data(), 0, sizeof(float) * dst_img.count());
// }

}  // namespace deepart
