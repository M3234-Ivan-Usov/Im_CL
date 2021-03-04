#include"im_executors.h"

#define HSx_CONVERTER 1
#define YCBCR_CONVERTER 2
#define CIE_CONVERTER -1

converser::converser(cl_context context, cl_command_queue queue, 
	functions* conversers) : executor(context, queue, conversers) {}

im_ptr converser::run(const std::string& from_cs, const std::string& to_cs, const im_object& src, int write_mode) {
	if (from_cs == "srgb") { return from_srgb(to_cs, src, write_mode); }
	else if (to_cs == "srgb") { return to_srgb(from_cs, src, write_mode); }
	else { return from_srgb(to_cs, *to_srgb(from_cs, src), write_mode); }
}

im_ptr converser::from_srgb(const std::string& colour_space, const im_object& src, int write_mode) {
	cl_kernel kern; int func = 0; float kr, kg, kb;
	if (colour_space == "hsv") {
		func = HSx_CONVERTER;  
		kern = kernels->at("srgb_to_hsv");
	}
	else if (colour_space == "hsl") {
		func = HSx_CONVERTER;  
		kern = kernels->at("srgb_to_hsl");
	}
	else if (colour_space == "ycc709") {
		func = YCBCR_CONVERTER; 
		kern = kernels->at("srgb_to_ycbcr");
		kr = 0.2126f, kg = 0.7152f, kb = 0.0722f;
	}
	else if (colour_space == "ycc601") {
		func = YCBCR_CONVERTER;  
		kern = kernels->at("srgb_to_ycbcr");
		kr = 0.299f, kg = 0.587f, kb = 0.114f;
	}
	else if (colour_space == "ycc2020") {
		func = YCBCR_CONVERTER;
		kern = kernels->at("srgb_to_ycbcr");
		kr = 0.2627f, kg = 0.678f, kb = 0.0593f;
	}
	else if (colour_space == "ciexyz") {
		func = CIE_CONVERTER; 
		kern = kernels->at("srgb_to_ciexyz");
	}
	else { throw std::runtime_error("Unknown colour space " + colour_space); }

	im_ptr dst = std::make_shared<im_object>(src.size, context, queue, write_mode);
	set_common_args(kern, src, *dst, write_mode);
	if (func == YCBCR_CONVERTER) { set_ycc_args(kern, &kr, &kg, &kb); }
	run_blocking(kern, src.size);
	return std::move(dst);
}

im_ptr converser::to_srgb(const std::string& colour_space, const im_object& src, int write_mode) {
	cl_kernel kern; int func = 0; float kr, kg, kb;
	if (colour_space == "hsv") {
		func = HSx_CONVERTER; 
		kern = kernels->at("hsv_to_srgb");
	}
	else if (colour_space == "hsl") {
		func = HSx_CONVERTER; 
		kern = kernels->at("hsl_to_srgb");
	}
	else if (colour_space == "ycc709") {
		func = YCBCR_CONVERTER;
		kern = kernels->at("ycbcr_to_srgb");
		kr = 0.2126f, kg = 0.7152f, kb = 0.0722f;
	}
	else if (colour_space == "ycc601") {
		func = YCBCR_CONVERTER; 
		kern = kernels->at("ycbcr_to_srgb");
		kr = 0.299f, kg = 0.587f, kb = 0.114f;
	}
	else if (colour_space == "ycc2020") {
		func = YCBCR_CONVERTER;
		kern = kernels->at("ycbcr_to_srgb");
		kr = 0.2627f, kg = 0.678f, kb = 0.0593f;
	}
	else if (colour_space == "ciexyz") {
		func = CIE_CONVERTER; 
		kern = kernels->at("ciexyz_to_srgb");
	}
	else { throw std::runtime_error("Unknown colour space " + colour_space); }

	im_ptr dst = std::make_shared<im_object>(src.size, context, queue, write_mode);
	set_common_args(kern, src, *dst, write_mode);
	if (func == YCBCR_CONVERTER) { set_ycc_args(kern, &kr, &kg, &kb); }
	run_blocking(kern, src.size);
	return std::move(dst);
}

void converser::set_ycc_args(cl_kernel kern, float* kr, float* kg, float* kb) {
	cl_int ret_code = clSetKernelArg(kern, 6, sizeof(float), kr);
	ret_code |= clSetKernelArg(kern, 7, sizeof(float), kg);
	ret_code |= clSetKernelArg(kern, 8, sizeof(float), kb);
	util::assert_success(ret_code, "Failed to set ycc params");
}
