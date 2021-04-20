#include"executor.h"

executor::executor(hardware* env, functions* kernels) : env(env), kernels(kernels) {}

cl_int executor::set_common_args(cl_kernel kern, cl_mem src, cl_sampler sampler, cl_mem dst) {
	cl_int ret_code = clSetKernelArg(kern, 0, sizeof(cl_mem), &src);
	ret_code |= clSetKernelArg(kern, 1, sizeof(cl_sampler), &sampler);
	ret_code |= clSetKernelArg(kern, 2, sizeof(cl_mem), &dst);
	return ret_code;
}

void executor::run_blocking(cl_kernel kern, cl_int2 size, cl_event* prev_event) {
	size_t global_size[2] = { (size_t)size.x, (size_t)size.y };
	cl_int ret_code = (prev_event == nullptr) ?
		clEnqueueNDRangeKernel(env->queue, kern, 2, NULL, global_size, NULL, 0, NULL, NULL) :
		clEnqueueNDRangeKernel(env->queue, kern, 2, NULL, global_size, NULL, 1, prev_event, NULL);
	ret_code |= clFinish(env->queue);
	util::assert_success(ret_code, "Failed to enqueue blocking kernel execution");
}


cl_event executor::run_with_event(cl_kernel kern, cl_int2 size, cl_event* prev_event) {
	size_t global_size[2] = { (size_t)size.x, (size_t)size.y }; cl_event next_event;
	cl_int ret_code = (prev_event == nullptr) ?
		clEnqueueNDRangeKernel(env->queue, kern, 2, NULL, global_size, NULL, 0, NULL, &next_event) :
		clEnqueueNDRangeKernel(env->queue, kern, 2, NULL, global_size, NULL, 1, prev_event, &next_event);
	util::assert_success(ret_code, "Failed to enqueue async kernel execution");
	return next_event;
}
