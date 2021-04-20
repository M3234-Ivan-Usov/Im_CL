#include"im_executors.h"

#define KERN_BILINEAR 0
#define KERN_LANCZOS 1
#define KERN_SPLINE 2
#define KERN_PRECISE 3

#define MITCHELL 0
#define CATMULL 1
#define ADOBE 2
#define B_SPLINE 3

#define FUNC 0
#define POL_INDEX 1
#define LAN_ORDER 2

zoomer::zoomer(hardware* env, functions* conv_kernels) : executor(env, conv_kernels) {
	/* Preallocate BC polynomials */
	auto pol_pair = calc_spline_polynom(1 / 3.0f, 1 / 3.0f);
	polynomials[MITCHELL][0] = pol_pair.first;
	polynomials[MITCHELL][1] = pol_pair.second;

	pol_pair = calc_spline_polynom(0.0f, 0.5f);
	polynomials[CATMULL][0] = pol_pair.first;
	polynomials[CATMULL][1] = pol_pair.second;
	
	pol_pair = calc_spline_polynom(0.0f, 0.75f);
	polynomials[ADOBE][0] = pol_pair.first;
	polynomials[ADOBE][1] = pol_pair.second;

	pol_pair = calc_spline_polynom(1.0f, 0.0f);
	polynomials[B_SPLINE][0] = pol_pair.first;
	polynomials[B_SPLINE][1] = pol_pair.second;
}

std::pair<cl_mem, cl_mem> zoomer::calc_spline_polynom(float B, float C) {
	float upper[4] = { 6.0f - 2.0f * B, 0.0f, -18.0f + 12.0f * B + 6.0f * C, 12.0f - 9.0f * B - 6.0f * C };
	float lower[4] = { 8.0f * B + 24.0f * C, -12.0f * B - 48.0f + C, 6.0f * B + 30.0f * C, -1.0f * B - 6.0f * C };
	return std::make_pair(
		env->alloc_buf(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 4 * sizeof(float), upper),
		env->alloc_buf(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 4 * sizeof(float), lower)
	);
}

zoomer::~zoomer() {
	for (size_t filter = 0; filter < 4; ++filter) {
		clReleaseMemObject(polynomials[filter][0]);
		clReleaseMemObject(polynomials[filter][1]);
	}
}

im_ptr zoomer::run(const std::string& kernel_type, float factor, im_ptr& src) {
	if (kernel_type == "precise") {
		cl_int new_x = static_cast<cl_int>(src->size.x * factor);
		cl_int new_y = static_cast<cl_int>(src->size.y * factor);
		return precise(src, {new_x, new_y});
	}
	int params[3] = { -1, -1, -1 };
	cl_kernel kern;

	if (kernel_type == "bilinear") { 
		params[FUNC] = KERN_BILINEAR; kern = kernels->at("bilinear");
		goto end_switch;
	}

	if (kernel_type == "lan3") { params[FUNC] = KERN_LANCZOS; params[LAN_ORDER] = 1; }
	else if (kernel_type == "lan4") { params[FUNC] = KERN_LANCZOS; params[LAN_ORDER] = 2; }
	else if (kernel_type == "lan5") { params[FUNC] = KERN_LANCZOS; params[LAN_ORDER] = 3; }

	if (params[FUNC] == KERN_LANCZOS) { kern = kernels->at("lanczos"); goto end_switch; }
	else if (kernel_type == "mitchell") { params[FUNC] = KERN_SPLINE;  params[POL_INDEX] = MITCHELL; }
	else if (kernel_type == "catmull") { params[FUNC] = KERN_SPLINE; params[POL_INDEX] = CATMULL; }
	else if (kernel_type == "adobe") { params[FUNC] = KERN_SPLINE; params[POL_INDEX] = ADOBE; }
	else if (kernel_type == "b-spline") { params[FUNC] = KERN_SPLINE; params[POL_INDEX] = B_SPLINE; }

	if (params[FUNC] == KERN_SPLINE) { kern = kernels->at("splines"); goto end_switch; }
	else  { throw std::runtime_error("Unknown kernel type " + kernel_type); }

	end_switch:
	bool upscale = factor >= 1.0f;
	float step_factor = upscale ? 2.0f : 0.5f;
	cl_int2 cur_size = src->size;

	cl_mem src_ptr = src->cl_storage;
	clRetainMemObject(src_ptr);
	cl_mem dst_ptr = nullptr;

	cl_sampler sampler = (params[FUNC] == KERN_BILINEAR) ?
		env->samplers.at({ CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_LINEAR }) :
		env->samplers.at({ CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_NEAREST });

	while ((upscale && factor > 2.0f) || (!upscale && factor < 0.5f)) {
		cur_size.x = static_cast<cl_int>(cur_size.x * step_factor);
		cur_size.y = static_cast<cl_int>(cur_size.y * step_factor);
		dst_ptr = env->alloc_im(cur_size);
		set_args(kern, src_ptr, dst_ptr, sampler, step_factor, params);
		run_blocking(kern, cur_size); std::swap(src_ptr, dst_ptr);
		clReleaseMemObject(dst_ptr);
		factor /= step_factor;
	}

	cur_size.x = static_cast<cl_int>(cur_size.x * factor);
	cur_size.y = static_cast<cl_int>(cur_size.y * factor);
	dst_ptr = env->alloc_im(cur_size);
	set_args(kern, src_ptr, dst_ptr, sampler, step_factor, params);
	run_blocking(kern, cur_size); clReleaseMemObject(src_ptr);
	return std::make_shared<im_object>(cur_size, env, dst_ptr);
}

void zoomer::set_args(cl_kernel kern, cl_mem src, cl_mem dst, 
	cl_sampler sampler, float factor, int* params) {
	cl_int ret_code = clSetKernelArg(kern, 0, sizeof(cl_mem), &src);
	ret_code |= clSetKernelArg(kern, 1, sizeof(cl_mem), &sampler);
	ret_code |= clSetKernelArg(kern, 2, sizeof(cl_mem), &dst);
	cl_float2 cl_fact = { factor, factor };
	ret_code |= clSetKernelArg(kern, 3, sizeof(cl_float2), &cl_fact);
	switch (params[FUNC]) {
	case KERN_BILINEAR:
		break;
	case KERN_LANCZOS:
		ret_code |= clSetKernelArg(kern, 4, sizeof(int), &params[LAN_ORDER]);
		break;
	case KERN_SPLINE:
		ret_code |= clSetKernelArg(kern, 4, sizeof(cl_mem), &polynomials[params[POL_INDEX]][0]);
		ret_code |= clSetKernelArg(kern, 5, sizeof(cl_mem), &polynomials[params[POL_INDEX]][1]);
		break;
	}
	util::assert_success(ret_code, "Failed to set extra args");

}

im_ptr zoomer::precise(im_ptr& src, cl_int2 new_size) {
	cl_kernel kern = kernels->at("precise");
	im_ptr result = std::make_shared<im_object>(new_size, env);
	cl_sampler sampler = env->samplers.at({ CL_ADDRESS_NONE, CL_FILTER_NEAREST });

	cl_int ret_code = clSetKernelArg(kern, 0, sizeof(cl_mem), &src->cl_storage);
	ret_code |= clSetKernelArg(kern, 1, sizeof(cl_sampler), &sampler);
	ret_code |= clSetKernelArg(kern, 2, sizeof(cl_mem), &result->cl_storage);

	int gcd_w = util::euclidean_gcd(src->size.x, new_size.x);
	int gcd_h = util::euclidean_gcd(src->size.y, new_size.y);
	cl_int2 split_out = { src->size.x / gcd_w, src->size.y / gcd_h };
	cl_int2 split_in = { new_size.x / gcd_w, new_size.y / gcd_h };
	cl_float area = 1.0f / (split_out.x * split_out.y);
	ret_code |= clSetKernelArg(kern, 3, sizeof(cl_int2), &split_out);
	ret_code |= clSetKernelArg(kern, 4, sizeof(cl_int2), &split_in);
	ret_code |= clSetKernelArg(kern, 5, sizeof(cl_float), &area);
	run_blocking(kern, new_size);
	return std::move(result);
}