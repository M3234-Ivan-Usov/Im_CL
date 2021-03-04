#include"im_executors.h"

#define USE_ALL_CHANNELS 3
#define USE_SINGLE_CHANNEL 0


contraster::contraster(cl_context context, cl_command_queue queue, functions* kernels,
	converser* converser_ptr) : executor(context, queue, kernels), converser_ptr(converser_ptr) {}


contraster::args::args(const std::string& t, const std::string& v,
	const std::string& c, const std::string& e, const std::string& r) {

	via_space = v.empty() ? "srgb" : v; type = t.empty() ? "exclusive" : t;
	if (type == "exclusive") { 
		exclude = e.empty()? 0.0039f : static_cast<float>(atof(e.c_str()) / 100);
	}
	else if (type == "adaptive") { 
		apt_radius = r.empty()? 1 : atoi(r.c_str()); 
		apt_exclude = e.empty() ? 1 : atoi(e.c_str());
	}
	else if (type == "manual") { 
		if (c.empty()) { throw std::runtime_error("Expected contrast param"); }
		contrast_level = static_cast<float>(atof(c.c_str()));
	}
}


im_ptr contraster::run(im_object& src, const args& params) {
	if (params.via_space == "srgb") { 
		if (params.type == "exclusive") { return exclusive_hist(src, params.exclude, USE_ALL_CHANNELS);}

		else if (params.type == "adaptive") { 
			return adaptive_hist(src, params.apt_radius, params.apt_exclude, USE_ALL_CHANNELS);
		}

		else if (params.type == "manual") { return manual(src, params.contrast_level, USE_ALL_CHANNELS); }

		else { throw std::runtime_error("Unknown type: " + params.type); }
	}

	else if (via_spaces.find(params.via_space) != via_spaces.end()) {
		im_ptr src_ptr = converser_ptr->run("srgb", params.via_space, src); im_ptr equalised = nullptr;

		if (params.type == "exclusive") { equalised = exclusive_hist(*src_ptr, params.exclude, USE_SINGLE_CHANNEL); }

		else if (params.type == "adaptive") {
			equalised = adaptive_hist(*src_ptr, params.apt_radius, params.apt_exclude, USE_SINGLE_CHANNEL);
		}

		else if (params.type == "manual") { equalised = manual(*src_ptr, params.contrast_level, USE_SINGLE_CHANNEL); }

		else { throw std::runtime_error("Unknown mode: " + params.type); }

		return converser_ptr->run(params.via_space, "srgb", *equalised);
	}
	else { throw std::runtime_error("Unknown colour space: " + params.via_space); }
}


im_ptr contraster::manual(im_object& src, float contrast, int channel_mode) {
	if (contrast < -1.0f || contrast > 1.0f) { 
		throw std::runtime_error("Invalid contrast, expected in range [-1..1]");
	}
	float c_val = contrast * 255.0f; int write_mode = WRITE_TO_BUFFER;
	c_val = (259.0f * (255.0f + c_val)) / (255.0f * (259.0f - c_val));
	cl_float4 contrast_4 = { c_val, c_val, c_val, 0.0f };
	if (channel_mode == USE_SINGLE_CHANNEL) {
		write_mode = NO_BUFFER_WRITE;
		contrast_4.y = 0.0f, contrast_4.z = 0.0f;
	}

	cl_kernel kern = kernels->at("manual");
	im_ptr dst = std::make_shared<im_object>(src.size, context, queue, write_mode);
	set_common_args(kern, src, *dst, write_mode);
	cl_int ret_code = clSetKernelArg(kern, 6, sizeof(cl_float4), &contrast_4);
	util::assert_success(ret_code, "Failed to set contrast arg");
	run_blocking(kern, src.size);
	return std::move(dst);
}


im_ptr contraster::exclusive_hist(im_object& src, float exclusive, int channel_mode) {
	int* hist = src.historgrams()[channel_mode];
	int mult = (channel_mode == USE_ALL_CHANNELS) ? 3 : 1;
	int exclude = static_cast<int>((mult * src.size.x * src.size.y) * exclusive);
	int min_val = 0, max_val = 255; int write_mode = WRITE_TO_BUFFER;

	for (int exclude_cnt = exclude; exclude_cnt > 0;
		exclude_cnt -= hist[min_val]) { ++min_val; }
	for (int exclude_cnt = exclude; exclude_cnt > 0;
		exclude_cnt -= hist[max_val]) { --max_val; }

	float off = min_val / 255.0f;
	float norm = (max_val - min_val) / 255.0f;
	cl_float4 off_4 = { off, off, off, 0.0f };
	cl_float4 norm_4 = { norm, norm, norm, 1.0f };
	if (channel_mode == USE_SINGLE_CHANNEL) {
		off_4.y = 0.0f, off_4.z = 0.0f;
		norm_4.y = 0.0f, norm_4.z = 0.0f;
		write_mode = NO_BUFFER_WRITE;
	}

	cl_kernel kern = kernels->at("exclusive_hist");
	im_ptr dst = std::make_shared<im_object>(src.size, context, queue, write_mode);
	set_common_args(kern, src, *dst, write_mode);
	cl_int ret_code = clSetKernelArg(kern, 6, sizeof(cl_float4), &off_4);
	ret_code |= clSetKernelArg(kern, 7, sizeof(cl_float4), &norm_4);
	util::assert_success(ret_code, "Failed to set extra args");
	run_blocking(kern, dst->size);
	return std::move(dst);
}


im_ptr contraster::adaptive_hist(im_object& src, int exclude, int radius, int channel_mode) {
	int lin_size = 2 * radius + 1;
	if (2 * exclude >= lin_size * lin_size) { throw std::runtime_error("To big exclusion for given radius"); }
	cl_kernel kern = nullptr; int write_mode = 0;
	if (channel_mode == USE_ALL_CHANNELS) { kern = kernels->at("adaptive_hist_all"); write_mode = WRITE_TO_BUFFER; }
	else { kern = kernels->at("adaptive_hist_single"); write_mode = NO_BUFFER_WRITE; }

	im_ptr dst = std::make_shared<im_object>(src.size, context, queue, write_mode);
	set_common_args(kern, src, *dst, write_mode);
	cl_int ret_code = clSetKernelArg(kern, 6, sizeof(int), &radius);
	ret_code |= clSetKernelArg(kern, 7, sizeof(int), &exclude);
	util::assert_success(ret_code, "Failed to set extra args");
	run_blocking(kern, dst->size);
	return dst;
}