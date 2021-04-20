#include"im_executors.h"

#define WAVELET_FORWARD 0
#define WAVELET_INVERSE 1

wavelet::wavelet(hardware* env, functions* wavelets) : executor(env, wavelets) {}

void wavelet::set_args(cl_kernel kern, cl_mem src,
	cl_sampler sampler, cl_mem dst, cl_int2 cur_size, int direction) {
	cl_int ret_code = clSetKernelArg(kern, 0, sizeof(cl_mem), &src);
	ret_code |= clSetKernelArg(kern, 1, sizeof(cl_sampler), &sampler);
	ret_code |= clSetKernelArg(kern, 2, sizeof(cl_mem), &dst);
	ret_code |= clSetKernelArg(kern, 3, sizeof(cl_int2), &cur_size);
	ret_code |= clSetKernelArg(kern, 4, sizeof(int), &direction);
}

im_ptr wavelet::run(const std::string& basis, float threshold, const im_ptr& src) {
	if (basis != "haar") { throw std::runtime_error("Unknowm wavelet basis: " + basis); }
	cl_int2 extended_size = { 1, 1 };
	while (extended_size.x < src->size.x) { extended_size.x <<= 1; }
	while (extended_size.y < src->size.y) { extended_size.y <<= 1; }

	cl_mem src_ptr = env->alloc_im(extended_size);
	cl_mem dst_ptr = env->alloc_im(extended_size);

	size_t origin[3] = { 0, 0, 0 };
	size_t region[3] = { (size_t)extended_size.x, (size_t)extended_size.y, 1 };

	cl_event q_event = nullptr;
	cl_int ret_code = clEnqueueCopyImage(env->queue, src->cl_storage,
		src_ptr, origin, origin, region, 0, nullptr, &q_event);

 	cl_sampler sampler = env->samplers.at({ CL_ADDRESS_CLAMP, CL_FILTER_NEAREST });
	cl_int2 cur_size = extended_size;

	cl_kernel kern = kernels->at("horizontal_" + basis);
	for (cur_size.x >>= 1; cur_size.x > 0; cur_size.x >>= 1) {
		set_args(kern, src_ptr, sampler, dst_ptr, cur_size, WAVELET_FORWARD);
		run_blocking(kern, { cur_size.x, extended_size.y }, &q_event);
		clEnqueueCopyImage(env->queue, dst_ptr, src_ptr, 
			origin, origin, region, 0, nullptr, &q_event);
		/*q_event = this->copy(*dst_ptr, *src_ptr,
			{ cur_size.x, 0 }, { cur_size.x, extended_size.y });*/
		std::swap(src_ptr, dst_ptr);
	}

	kern = kernels->at("vertical_" + basis);
	cur_size.x = extended_size.x;
	for (cur_size.y >>= 1; cur_size.y > 0; cur_size.y >>= 1) {
		set_args(kern, src_ptr, sampler, dst_ptr, cur_size, WAVELET_FORWARD);
		run_blocking(kern, { extended_size.x, cur_size.y }, &q_event);
		clEnqueueCopyImage(env->queue, dst_ptr, src_ptr,
			origin, origin, region, 0, nullptr, &q_event);
		/*q_event = this->copy(*dst_ptr, *src_ptr,
			{ 0, cur_size.y }, { extended_size.x, cur_size.y});*/
		std::swap(src_ptr, dst_ptr);
	}

	kern = kernels->at("soft_threshold");
	//set_args(kern, *src_ptr, *dst_ptr);
	clSetKernelArg(kern, 6, sizeof(float), &threshold);
	run_blocking(kern, { extended_size.x, extended_size.y }, &q_event);
	std::swap(src_ptr, dst_ptr); q_event = nullptr;

	kern = kernels->at("vertical_" + basis);
	for (cur_size.y = 1; cur_size.y < extended_size.y; cur_size.y <<= 1) {
		set_args(kern, src_ptr, sampler, dst_ptr, cur_size, WAVELET_INVERSE);
		run_blocking(kern, { extended_size.x, cur_size.y }, &q_event);
		clEnqueueCopyImage(env->queue, dst_ptr, src_ptr,
			origin, origin, region, 0, nullptr, &q_event);
	/*	q_event = this->copy(*dst_ptr, *src_ptr,
			{ 0, cur_size.y }, { extended_size.x, cur_size.y });*/
		std::swap(src_ptr, dst_ptr);
	}

	kern = kernels->at("horizontal_" + basis);
	for (cur_size.x = 1; cur_size.x < extended_size.x; cur_size.x <<= 1) {
		set_args(kern, src_ptr, sampler, dst_ptr, cur_size, WAVELET_INVERSE);
		run_blocking(kern, { extended_size.x, cur_size.y }, &q_event);
		clEnqueueCopyImage(env->queue, dst_ptr, src_ptr,
			origin, origin, region, 0, nullptr, &q_event);
		/*q_event = this->copy(*dst_ptr, *src_ptr,
			{ 0, cur_size.y }, { extended_size.x, cur_size.y });*/
		std::swap(src_ptr, dst_ptr);
	}

	im_ptr result = std::make_shared<im_object>(src->size, env);
	region[0] = src->size.x, region[1] = src->size.y;
	clEnqueueCopyImage(env->queue, src_ptr, result->cl_storage,
		origin, origin, region, 0, nullptr, nullptr);
	clFinish(env->queue);
	clReleaseMemObject(src_ptr); clReleaseMemObject(dst_ptr);
	return std::move(result);
}