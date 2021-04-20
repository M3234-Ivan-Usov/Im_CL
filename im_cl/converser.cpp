#include"im_executors.h"

#define HSx_CONVERTER 1
#define YCBCR_CONVERTER 2
#define CIE_CONVERTER -1

converser::converser(hardware* env, functions* conversers) : executor(env, conversers) {}

std::unordered_map<std::string, std::set<std::string>> converser::conversions = {
	{"srgb", {"ycc601", "ycc709", "ycc2020", "hsl", "hsv", "ciexyz"}},
	{"ycc601", {"srgb"}}, {"ycc709", {"srgb"}}, {"ycc2020", {"srgb"}},
	{"hsl", {"srgb", "hsv"}}, {"hsv", {"srgb", "hsl"}},
	{"ciexyz", {"srgb", "cielab"}}, {"cielab", {"ciexyz"}}
};

std::unordered_map<std::string, cl_float3> converser::ycc_params = {
	{"ycc601", { 0.299f, 0.587f, 0.114f }},
	{"ycc709", { 0.2126f, 0.7152f, 0.0722f }},
	{"ycc2020", { 0.2627f, 0.678f, 0.0593f }},
};

im_ptr converser::run(col_pair colours, im_ptr& src) {
	auto it_src = conversions.find(colours.first);
	if (it_src == conversions.end()) { 
		std::string error_message = "Unknown input colour space: "
			+ colours.first + "\nAvaliable:";
		for (const auto& x : conversions) { 
			error_message.append(" ").append(x.first);
		}
		throw std::runtime_error(error_message);
	}
	auto it_dst = it_src->second.find(colours.second);
	if (it_dst == it_src->second.end()) {
		std::string error_message = "Unknown output colour space: "
			+ colours.second + "\nAvaliable:";
		for (const auto& x : it_src->second) { 
			error_message.append(" ").append(x);
		}
		throw std::runtime_error(error_message);
	}

	bool set_extra_args = false;
	auto ycc_it = ycc_params.find(colours.first);
	if (ycc_it == ycc_params.end()) {
		ycc_it = ycc_params.find(colours.second);
		if (ycc_it != ycc_params.end()) {
			set_extra_args = true;
			colours.second = "ycbcr";
		}
	}
	else {
		set_extra_args = true;
		colours.first = "ycbcr";
	}
	cl_kernel kern = kernels->at(colours.first + "_to_" + colours.second);
	im_ptr dst = std::make_shared<im_object>(src->size, env);
	set_args(kern, src, dst);
	if (set_extra_args) { clSetKernelArg(kern, 3, sizeof(cl_float3), &ycc_it->second); }
	run_blocking(kern, src->size);
	return std::move(dst);
}

void converser::set_args(cl_kernel kern, im_ptr& src, im_ptr& dst) {
	cl_sampler sampler = env->samplers.at({ CL_ADDRESS_NONE, CL_FILTER_NEAREST });
	clSetKernelArg(kern, 0, sizeof(cl_mem), &src->cl_storage);
	clSetKernelArg(kern, 1, sizeof(cl_sampler), &sampler);
	clSetKernelArg(kern, 2, sizeof(cl_mem), &dst->cl_storage);
}