#pragma once
#include<CL/cl.h>
#include<string>
#include<map>

struct hardware {
	cl_platform_id* platforms = nullptr;
	cl_platform_id cur_platform = nullptr;
	cl_device_id cur_device = nullptr;
	size_t dev_num = 0, plat_num = 0;
	size_t platform_id = 0, device_id = 0;

	cl_command_queue queue;
	cl_context context;

	cl_mem preallocation = nullptr;
	size_t prealloc_size = 0;

	using sampler_params = std::pair<cl_addressing_mode, cl_filter_mode>;
	std::map<sampler_params, cl_sampler> samplers;

	hardware() = default;
	hardware(size_t platform_id, size_t device_id, size_t prealloc_size);

	template<typename target_value>
	static target_value device_param(cl_device_id device, cl_device_info param);
	static std::string string_param(cl_device_id device, cl_device_info param);

	static void env_info();
	static void device_info(cl_device_id device, bool extensions = false);

	cl_mem alloc_buf(cl_mem_flags flags, size_t size, void* ptr);
	cl_mem alloc_im(cl_int2 size, float* ptr = nullptr, cl_uint type = CL_RGBA);

	~hardware();
};