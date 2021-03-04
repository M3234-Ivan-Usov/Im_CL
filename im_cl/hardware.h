#pragma once
#include<CL/cl.h>
#include<string>

struct hardware {
	cl_platform_id* platforms = nullptr;
	cl_platform_id cur_platform = nullptr;
	cl_device_id cur_device = nullptr;
	size_t dev_num = 0, plat_num = 0;
	size_t platform_id = 0, device_id = 0;

	hardware() = default;
	hardware(size_t platform_id, size_t device_id);

	template<typename target_value>
	static target_value device_param(cl_device_id device, cl_device_info param);
	static std::string string_param(cl_device_id device, cl_device_info param);

	static void env_info();
	static void device_info(cl_device_id device);

	~hardware();
};