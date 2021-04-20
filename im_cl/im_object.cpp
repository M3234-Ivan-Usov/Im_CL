#include"im_object.h"
#include"util.h"


im_object::im_object(cl_int2 size, hardware* env, cl_mem storage) : size(size),
	alloc_size(3 * size.x * size.y), env(env), host_ptr(nullptr) {
	if (storage != nullptr) { this->cl_storage = storage; }
	else { cl_storage = env->alloc_im(size); }
}

im_object::im_object(char* host_ptr, size_t width, size_t height, hardware* env, int direct_gamma) : 
	host_ptr(host_ptr), env(env), alloc_size(3 * width * height) {
	this->size = { (cl_int)width, (cl_int)height };

	cl_mem temp_buf = (alloc_size > env->prealloc_size) ?
		env->alloc_buf(CL_MEM_READ_ONLY, alloc_size, nullptr) : env->preallocation;
	cl_event copy_event = nullptr;
	cl_int ret_code = clEnqueueWriteBuffer(env->queue, temp_buf,
		CL_FALSE, 0, alloc_size, host_ptr, 0, nullptr, &copy_event);
	cl_kernel norm_kern = util::kernels->at("normalise");
	cl_storage = env->alloc_im(size);
	ret_code = clSetKernelArg(norm_kern, 0, sizeof(cl_mem), &temp_buf);
	ret_code |= clSetKernelArg(norm_kern, 1, sizeof(cl_int2), &size);
	ret_code |= clSetKernelArg(norm_kern, 2, sizeof(cl_mem), &cl_storage);
	ret_code |= clSetKernelArg(norm_kern, 3, sizeof(cl_int), &direct_gamma);

	size_t global_size[2] = { (size_t)size.x, (size_t)size.y };
	ret_code = clEnqueueNDRangeKernel(env->queue, norm_kern,
		2, NULL, global_size, NULL, 0, NULL, NULL);
	ret_code |= clFinish(env->queue);
	if (alloc_size > env->prealloc_size) { clReleaseMemObject(temp_buf); }
}

im_object::im_object(im_object&& other) noexcept : cl_storage(other.cl_storage),
	env(other.env), size(other.size), alloc_size(other.alloc_size) {
	if (cl_storage != nullptr) { clRetainMemObject(cl_storage); }
	delete[] host_ptr;
	host_ptr = other.host_ptr;
	other.host_ptr = nullptr;
}

char* im_object::get_host_ptr(int inverse_gamma) {
	if (host_ptr == nullptr) {
		host_ptr = new char[alloc_size];
		cl_mem temp_buf = (alloc_size > env->prealloc_size) ?
			env->alloc_buf(CL_MEM_WRITE_ONLY, alloc_size, nullptr) : env->preallocation;

		cl_kernel kern = util::kernels->at("denormalise");
		cl_int ret_code = clSetKernelArg(kern, 0, sizeof(cl_mem), &cl_storage);
		ret_code |= clSetKernelArg(kern, 1, sizeof(cl_sampler), &env->samplers.at({ CL_ADDRESS_NONE, CL_FILTER_NEAREST }));
		ret_code |= clSetKernelArg(kern, 2, sizeof(cl_mem), &temp_buf);
		ret_code |= clSetKernelArg(kern, 3, sizeof(cl_int2), &size);
		ret_code |= clSetKernelArg(kern, 4, sizeof(cl_int), &inverse_gamma);

		size_t global_size[2] = { (size_t)size.x, (size_t)size.y };
		ret_code = clEnqueueNDRangeKernel(env->queue, kern,
			2, NULL, global_size, NULL, 0, NULL, NULL);
		clFinish(env->queue);
		ret_code |= clEnqueueReadBuffer(env->queue, temp_buf,
			CL_TRUE, 0, alloc_size, host_ptr, 0, NULL, NULL);
		util::assert_success(ret_code, "Failed to read from device");
		if (alloc_size > env->prealloc_size) { clReleaseMemObject(temp_buf); }
	}
	return host_ptr;
}

channels im_object::get_channels(int inverse_gamma) {
	cl_mem temp_buf = (alloc_size > env->prealloc_size) ?
		env->alloc_buf(CL_MEM_WRITE_ONLY, alloc_size, nullptr) : env->preallocation;

	cl_kernel kern = util::kernels->at("denormalise");
	cl_sampler sampler = env->samplers.at({ CL_ADDRESS_NONE, CL_FILTER_NEAREST });
	cl_int ret_code = clSetKernelArg(kern, 0, sizeof(cl_mem), &cl_storage);
	ret_code |= clSetKernelArg(kern, 1, sizeof(cl_sampler), &sampler);
	ret_code |= clSetKernelArg(kern, 2, sizeof(cl_mem), &temp_buf);
	ret_code |= clSetKernelArg(kern, 3, sizeof(cl_int2), &size);
	ret_code |= clSetKernelArg(kern, 4, sizeof(cl_int), &inverse_gamma);

	size_t global_size[2] = { (size_t)size.x, (size_t)size.y };
	cl_event q_event = nullptr;
	ret_code = clEnqueueNDRangeKernel(env->queue, kern,
		2, NULL, global_size, NULL, 0, NULL, &q_event);

	size_t channel_size = size.x * size.y;
	channels host_channels = { new char[channel_size], new char[channel_size], new char[channel_size] };
	for (size_t channel = 0; channel < 2; ++channel) {
		ret_code |= clEnqueueReadBuffer(env->queue, temp_buf, CL_FALSE,
			channel_size * channel, channel_size, host_channels[channel], 1, &q_event, NULL);
	}
	util::assert_success(ret_code, "Failed to read from device");
	if (alloc_size > env->prealloc_size) { clReleaseMemObject(temp_buf); }
	return host_channels;
}



histogram im_object::calc_histograms(int inverse_gamma) {
	auto src = reinterpret_cast<unsigned char*>(get_host_ptr(inverse_gamma));
	histogram hist;
	// hist.fill(std::array<int, 256>());
	for (size_t channel = 0; channel < 4; ++channel) {
		hist[channel].fill(0);
	}
	for (size_t pix = 0; pix < alloc_size; pix += 3) {
		for (size_t col = 0; col < 3; ++col) {
			int val = static_cast<int>(src[pix + col]);
			hist[col][val]++; hist[3][val]++;
		}
	}
	return hist;
}

im_object::~im_object() {
	if (cl_storage != nullptr) { clReleaseMemObject(cl_storage); }
	delete[] host_ptr;
}