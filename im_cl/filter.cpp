#include"im_executors.h"

filter::filter(cl_context context, cl_command_queue queue, 
	functions* filters) : executor(context, queue, filters) {}

im_ptr filter::gauss(float sigma, int lin_size, im_object& src) {
	float divisor = sqrtf(2.0f * CL_M_PI_F) * sigma;
	float exp_divisor = 2.0f * sigma * sigma;
	int radius = (lin_size - 1) / 2; cl_int ret_code;
	float* conv_kern = new float[lin_size];
	for (int w = -radius; w <= radius; ++w) {
		conv_kern[radius + w] = expf(-(w * w) / exp_divisor) / divisor;
	}
	cl_mem kern_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) * lin_size, conv_kern, &ret_code);
	util::assert_success(ret_code, "Failed to create buffer");

	cl_kernel kern = kernels->at("horizontal_conv");
	im_object temp(src.size, context, queue, NO_BUFFER_WRITE);
	set_common_args(kern, src, temp, NO_BUFFER_WRITE);
	ret_code = clSetKernelArg(kern, 6, sizeof(int), &radius);
	ret_code |= clSetKernelArg(kern, 7, sizeof(cl_mem), &kern_buf);
	cl_event horizontal = run_with_event(kern, temp.size);

	kern = kernels->at("vertical_conv");
	im_ptr result = std::make_shared<im_object>(temp.size, context, queue, WRITE_TO_BUFFER);
	set_common_args(kern, temp, *result, WRITE_TO_BUFFER);
	ret_code = clSetKernelArg(kern, 6, sizeof(int), &radius);
	ret_code |= clSetKernelArg(kern, 7, sizeof(cl_mem), &kern_buf);

	run_blocking(kern, temp.size, &horizontal);
	clReleaseMemObject(kern_buf); delete[] conv_kern;
	return std::move(result);
}