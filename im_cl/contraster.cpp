#include"im_executors.h"


contraster::contraster(hardware* env, functions* kernels) : executor(env, kernels) {}

void contraster::set_args(cl_kernel kern, const im_ptr& src, im_ptr& dst) {
	cl_int ret_code = clSetKernelArg(kern, 0, sizeof(cl_mem), src->cl_storage);
	ret_code |= clSetKernelArg(kern, 1, sizeof(cl_int2), &src->size);
	ret_code |= clSetKernelArg(kern, 2, sizeof(cl_mem), dst->cl_storage);
}


im_ptr contraster::manual(im_ptr& src, float contrast, int channel_mode) {
	if (contrast < -1.0f || contrast > 1.0f) { 
		throw std::runtime_error("Invalid contrast, expected in range [-1..1]");
	}
	float c_val = contrast * 255.0f;
	c_val = (259.0f * (255.0f + c_val)) / (255.0f * (259.0f - c_val));
	cl_float4 contrast_vec = { c_val, 0.0f, 0.0f, 1.0f };
	if (channel_mode == all_channels) {
		contrast_vec.y = c_val, contrast_vec.z = c_val;
	}
	cl_kernel kern = kernels->at("manual");
	im_ptr dst = std::make_shared<im_object>(src->size, env);
	cl_sampler sampler = env->samplers.at({ CL_ADDRESS_NONE, CL_FILTER_NEAREST });
	cl_int ret_code = set_common_args(kern, src->cl_storage, sampler, dst->cl_storage);
	ret_code |= clSetKernelArg(kern, 3, sizeof(cl_float4), &contrast_vec);
	run_blocking(kern, src->size);
	return std::move(dst);
}


im_ptr contraster::exclusive_hist(im_ptr& src, float exclusive, int channel_mode) {
	int* hist = src->calc_histograms(GAMMA_CORRECTION_OFF)[channel_mode].data();
	int mult = (channel_mode == all_channels) ? 3 : 1;
	int exclude = static_cast<int>((mult * src->size.x * src->size.y) * exclusive);
	int min_val = 0, max_val = 255;

	for (int exclude_cnt = exclude; exclude_cnt > 0;
		exclude_cnt -= hist[min_val]) { ++min_val; }
	for (int exclude_cnt = exclude; exclude_cnt > 0;
		exclude_cnt -= hist[max_val]) { --max_val; }

	if (min_val >= max_val) { throw std::runtime_error("Too much exclusive"); }
	float off = min_val / 255.0f, norm = (max_val - min_val) / 255.0f;
	cl_float4 off_vec = { off, off, off, 0.0f };
	cl_float4 norm_vec = { norm, norm, norm, 1.0f };
	if (channel_mode == single_channel) {
		off_vec.y = 0.0f, off_vec.z = 0.0f;
		norm_vec.y = 1.0f, norm_vec.z = 1.0f;
	}

	cl_kernel kern = kernels->at("exclusive_hist");
	im_ptr dst = std::make_shared<im_object>(src->size, env);
	cl_sampler sampler = env->samplers.at({ CL_ADDRESS_NONE, CL_FILTER_NEAREST });
	cl_int ret_code = set_common_args(kern, src->cl_storage, sampler, dst->cl_storage);
	ret_code |= clSetKernelArg(kern, 3, sizeof(cl_float4), &off_vec);
	ret_code |= clSetKernelArg(kern, 4, sizeof(cl_float4), &norm_vec);
	run_blocking(kern, dst->size);
	return std::move(dst);
}


im_ptr contraster::adaptive_hist(im_ptr& src, cl_int2 region, int exclude, int channel_mode) {
	if (2 * exclude >= region.x * region.y) { 
		throw std::runtime_error("To big exclusion for given region");
	}
	cl_kernel kern = kernels->at("adaptive_hist");
	im_ptr dst = std::make_shared<im_object>(src->size, env);
	cl_sampler sampler = env->samplers.at({ CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_NEAREST });
	cl_int ret_code = set_common_args(kern, src->cl_storage, sampler, dst->cl_storage);
	ret_code |= clSetKernelArg(kern, 3, sizeof(cl_int2), &src->size);
	ret_code |= clSetKernelArg(kern, 4, sizeof(cl_int2), &region);
	ret_code |= clSetKernelArg(kern, 5, sizeof(int), &exclude);
	int x_region = src->size.x / region.x + (src->size.x % region.x == 0) ? 0 : 1;
	int y_region = src->size.y / region.y + (src->size.y % region.y == 0) ? 0 : 1;
	run_blocking(kern, { x_region, y_region });
	return dst;
}