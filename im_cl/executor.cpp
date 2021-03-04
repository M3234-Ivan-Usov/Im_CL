#include"executor.h"

executor::executor(cl_context context, cl_command_queue queue, functions* kernels) :
	context(context), queue(queue), kernels(kernels) {}


void executor::set_common_args(cl_kernel kern,
	const im_object& src, const im_object& dst, int write_mode) {

	cl_int ret_code = clSetKernelArg(kern, 0, sizeof(cl_mem), &src.cl_im);
	util::assert_success(ret_code, "Failed to set src image arg");

	ret_code = clSetKernelArg(kern, 1, sizeof(cl_mem), &dst.cl_storage);
	util::assert_success(ret_code, "Failed to set buffer arg");

	ret_code = clSetKernelArg(kern, 2, sizeof(cl_mem), &dst.cl_im);
	util::assert_success(ret_code, "Failed to set dst image arg");

	ret_code = clSetKernelArg(kern, 3, sizeof(cl_sampler), &dst.sampler);
	util::assert_success(ret_code, "Failed to set sampler arg");

	ret_code = clSetKernelArg(kern, 4, sizeof(cl_int2), &dst.size);
	util::assert_success(ret_code, "Failed to set out size arg");

	ret_code = clSetKernelArg(kern, 5, sizeof(int), &write_mode);
	util::assert_success(ret_code, "Failed to set write mode arg");
}


void executor::run_blocking(cl_kernel kern, cl_int2 size, cl_event* prev_event) {
	size_t global_size[2] = { (size_t)size.x, (size_t)size.y };
	cl_int ret_code = (prev_event == nullptr) ?
		clEnqueueNDRangeKernel(queue, kern, 2, NULL, global_size, NULL, 0, NULL, NULL) :
		clEnqueueNDRangeKernel(queue, kern, 2, NULL, global_size, NULL, 1, prev_event, NULL);
	util::assert_success(ret_code, "Failed to enqueue blocking kernel execution"); clFinish(queue);
}


cl_event executor::run_with_event(cl_kernel kern, cl_int2 size, cl_event* prev_event) {
	size_t global_size[2] = { (size_t)size.x, (size_t)size.y }; cl_event next_event;
	cl_int ret_code = (prev_event == nullptr) ?
		clEnqueueNDRangeKernel(queue, kern, 2, NULL, global_size, NULL, 0, NULL, &next_event) :
		clEnqueueNDRangeKernel(queue, kern, 2, NULL, global_size, NULL, 1, prev_event, &next_event);
	util::assert_success(ret_code, "Failed to enqueue async kernel execution");
	return next_event;
}