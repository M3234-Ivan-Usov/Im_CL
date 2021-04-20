#include"im_executors.h"

filter::filter(hardware* env, functions* filters) : executor(env, filters) {}

im_ptr filter::gauss(float sigma, int lin_size, im_ptr& src) {
	float divisor = -2.0f * sigma * sigma;
	float pi_div = 2.0f * sigma * sigma * CL_M_PI;
	cl_int radius = (lin_size - 1) / 2;
	float* conv_kern = new float[lin_size * lin_size];
	for (int y = -radius, p = 0; y <= radius; ++y) {
		for (int x = -radius; x <= radius; ++x, ++p) {
			conv_kern[p] = expf((x * x + y * y) / divisor) / pi_div;
		}
	}
	cl_mem im_kernel = env->alloc_im({ lin_size, lin_size }, conv_kern, CL_A);
	im_ptr result = convolve(im_kernel, src, radius);
	clReleaseMemObject(im_kernel);
	delete[] conv_kern;
	return std::move(result);
}

im_ptr filter::convolve(cl_mem conv_kern, im_ptr& src, cl_int radius) {
	cl_kernel kern = kernels->at("conv_2D");
	im_ptr result = std::make_shared<im_object>(src->size, env);
	cl_sampler sampler = env->samplers.at({ CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_NEAREST });
	cl_int ret_code = set_common_args(kern, src->cl_storage, sampler, result->cl_storage);
	ret_code |= clSetKernelArg(kern, 3, sizeof(cl_int), &radius);
	ret_code |= clSetKernelArg(kern, 4, sizeof(cl_mem), &conv_kern);
	run_blocking(kern, src->size);
	return std::move(result);
}