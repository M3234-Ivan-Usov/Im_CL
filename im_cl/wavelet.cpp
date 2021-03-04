#include"im_executors.h"

#define WAVELET_HORIZONTAL 0
#define WAVELET_VERTICAL 1

wavelet::wavelet(cl_context context, cl_command_queue queue, functions* wavelets, cl_kernel gamma_corrector) :
	executor(context, queue, wavelets), gamma_corrector(gamma_corrector) { cl_int ret_code;
	nullable_sampler = clCreateSampler(context, CL_FALSE, CL_ADDRESS_CLAMP, CL_FILTER_NEAREST, &ret_code);
	util::assert_success(ret_code, "Failed to create wavelet sampler");
}

wavelet::~wavelet() { clReleaseSampler(nullable_sampler); }

void wavelet::set_haar_args(cl_kernel kern, im_object* src, im_object* dst_next, cl_int2 cur_size, int dim) {
	cl_int ret_code = clSetKernelArg(kern, 0, sizeof(cl_mem), src);
	util::assert_success(ret_code, "Failed to set src image");
	ret_code = clSetKernelArg(kern, 1, sizeof(cl_mem), dst_next);
	util::assert_success(ret_code, "Failed to set dst image");
	ret_code = clSetKernelArg(kern, 2, sizeof(cl_sampler), &nullable_sampler);
	util::assert_success(ret_code, "Failed to set sampler");
	ret_code = clSetKernelArg(kern, 3, sizeof(cl_int2), &cur_size);
	util::assert_success(ret_code, "Failed to set current size");
	ret_code = clSetKernelArg(kern, 4, sizeof(int), &dim);
	util::assert_success(ret_code, "Failed to set wavelet dim");
}

im_ptr wavelet::run(const std::string& basis, float threshold, const im_object& src) {
	if (basis != "haar") { throw std::runtime_error("Unknowm wavelet basis: " + basis); }

	cl_int2 extended_size = { (cl_int)util::next_power(src.size.x), (cl_int)util::next_power(src.size.y) };
	cl_int2 cur_size = extended_size; cl_event* q_event = nullptr;
	size_t origin[3] = { 0, 0, 0 }, region[3] = { (size_t)src.size.x, (size_t)src.size.y, 1 };

	im_object* src_ptr = new im_object(cur_size, context, queue, NO_BUFFER_WRITE);
	cl_int ret_code = clEnqueueCopyImage(queue, src.cl_im, 
		src_ptr->cl_im, origin, origin, region, 0, NULL, q_event);
	util::assert_success(ret_code, "Failed to copy source");
	im_object* dst_ptr = new im_object(cur_size, context, queue, NO_BUFFER_WRITE);

	region[0] = extended_size.x, region[1] = extended_size.y;
	cl_kernel kern = kernels->at("direct_haar");
	for (cur_size.x >>= 1; cur_size.x > 0; cur_size.x >>= 1) {
		set_haar_args(kern, src_ptr, dst_ptr, cur_size, WAVELET_HORIZONTAL);
		run_blocking(kern, { cur_size.x, extended_size.y }, q_event);
		std::swap(src_ptr, dst_ptr);
		ret_code = clEnqueueCopyImage(queue, src_ptr->cl_im,
			dst_ptr->cl_im,origin, origin, region, 0, NULL, q_event);
		util::assert_success(ret_code, "Failed to copy while direct pass");
	}

	region[0] = src.size.x, cur_size.x = extended_size.x;
	for (cur_size.y >>= 1; cur_size.y > 0; cur_size.y >>= 1) {
		set_haar_args(kern, src_ptr, dst_ptr, cur_size, WAVELET_VERTICAL);
		run_blocking(kern, { extended_size.x, cur_size.y }, q_event);
		std::swap(src_ptr, dst_ptr);
		ret_code =clEnqueueCopyImage(queue, src_ptr->cl_im,
			dst_ptr->cl_im, origin, origin, region, 0, NULL, q_event);
		util::assert_success(ret_code, "Failed to copy while direct pass");
	}

	kern = kernels->at("soft_threshold");
	set_common_args(kern, *src_ptr, *dst_ptr, NO_BUFFER_WRITE);
	clSetKernelArg(kern, 6, sizeof(float), &threshold);
	run_blocking(kern, { extended_size.x, extended_size.y }, q_event);
	std::swap(src_ptr, dst_ptr); q_event = nullptr;

	kern = kernels->at("inverse_haar");
	for (cur_size.y = 1; cur_size.y < extended_size.y; cur_size.y <<= 1) {
		set_haar_args(kern, src_ptr, dst_ptr, cur_size, WAVELET_VERTICAL);
		run_blocking(kern, { extended_size.x, cur_size.y }, q_event);
		std::swap(src_ptr, dst_ptr);
		ret_code = clEnqueueCopyImage(queue, src_ptr->cl_im, 
			dst_ptr->cl_im, origin, origin, region, 0, NULL, q_event);
		util::assert_success(ret_code, "Failed to copy while inverse pass");
	}

	for (cur_size.x = 1; cur_size.x < extended_size.x; cur_size.x <<= 1) {
		set_haar_args(kern, src_ptr, dst_ptr, cur_size, WAVELET_HORIZONTAL);
		run_blocking(kern, { cur_size.x, extended_size.y }, q_event);
		std::swap(dst_ptr, src_ptr);
		ret_code = clEnqueueCopyImage(queue, src_ptr->cl_im, 
			dst_ptr->cl_im, origin, origin, region, 0, NULL, q_event);
		util::assert_success(ret_code, "Failed to copy while inverse pass");
	}

	im_ptr result = std::make_shared<im_object>(src.size, context, queue, WRITE_TO_BUFFER);
	set_common_args(gamma_corrector, *src_ptr, *result, WRITE_TO_BUFFER);
	run_blocking(gamma_corrector, src.size);
	delete src_ptr; delete dst_ptr;
	return std::move(result);
}