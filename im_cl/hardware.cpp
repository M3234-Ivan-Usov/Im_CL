#include"hardware.h"
#include"util.h"


hardware::hardware(size_t platform_id, size_t device_id) :
	platform_id(platform_id), device_id(device_id) {
	clGetPlatformIDs(0, NULL, &plat_num);
	if (platform_id >= plat_num) { throw std::runtime_error("Illegal platform"); }
	platforms = new cl_platform_id[plat_num];
	clGetPlatformIDs(plat_num, platforms, NULL);
	cl_int ret_code = clGetPlatformIDs(plat_num, platforms, NULL);
	util::assert_success(ret_code, "Failed to get platforms");
	cur_platform = platforms[platform_id];
	clGetDeviceIDs(platforms[platform_id], CL_DEVICE_TYPE_GPU, 0, NULL, &dev_num);
	if (device_id >= dev_num) { throw std::runtime_error("Illegal device"); }
	cl_device_id* devices = new cl_device_id[dev_num];
	clGetDeviceIDs(platforms[platform_id], CL_DEVICE_TYPE_GPU, dev_num, devices, NULL);
	util::assert_success(ret_code, "Failed to get devices");
	cur_device = devices[device_id];
	for (size_t i = 0; i < dev_num; ++i) { if (i != device_id) { clReleaseDevice(devices[i]); } }
	delete[] devices;
}


template<typename target_value>
target_value hardware::device_param(cl_device_id device, cl_device_info param) {
	target_value result; cl_uint ret_size;
	clGetDeviceInfo(device, param, sizeof(target_value), &result, &ret_size);
	return result;
}

std::string hardware::string_param(cl_device_id device, cl_device_info param) {
	char dev_param[256]; size_t ret_size;
	clGetDeviceInfo(device, param, 256, dev_param, &ret_size);
	return std::string(dev_param, ret_size);
}

void hardware::env_info() {
	char* env_str = new char[4096];
	cl_uint plat_num, dev_num;
	size_t ret_size;
	clGetPlatformIDs(0, NULL, &plat_num);
	cl_platform_id* plat_ids = new cl_platform_id[plat_num];
	clGetPlatformIDs(plat_num, plat_ids, NULL);
	for (size_t pl_id = 0; pl_id < plat_num; ++pl_id) {
		clGetPlatformInfo(plat_ids[pl_id], CL_PLATFORM_NAME, 4096, env_str, &ret_size);
		std::cout << "Platform " << pl_id << ": " << std::string(env_str, ret_size) << std::endl;
		clGetDeviceIDs(plat_ids[pl_id], CL_DEVICE_TYPE_GPU, 0, NULL, &dev_num);
		cl_device_id* dev_ids = new cl_device_id[dev_num];
		clGetDeviceIDs(plat_ids[pl_id], CL_DEVICE_TYPE_GPU, dev_num, dev_ids, NULL);
		for (size_t d_id = 0; d_id < dev_num; ++d_id) {
			clGetDeviceInfo(dev_ids[d_id], CL_DEVICE_NAME, 4096, env_str, &ret_size);
			std::cout << "  Device " << d_id << ": " << std::string(env_str, ret_size) <<
				" (" << string_param(dev_ids[d_id], CL_DEVICE_VERSION) << ")" << std::endl;
			clReleaseDevice(dev_ids[d_id]);
		}
		delete[] dev_ids;
	}
	delete[] plat_ids, env_str;
}

void hardware::device_info(cl_device_id device) {
	float kb = 1024.0f, mb = 1024.0f * 1024.0f;
	std::cout << "< " << string_param(device, CL_DEVICE_NAME) << " (" <<
		string_param(device, CL_DEVICE_VERSION) << ") >" << std::endl;
	std::cout << "Global cache size:      " << device_param<cl_ulong>(device,
		CL_DEVICE_GLOBAL_MEM_CACHE_SIZE) / kb << " KBs" << std::endl;
	std::cout << "Global cache-line size: " << device_param<cl_uint>(device,
		CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE) << " bytes" << std::endl;
	std::cout << "Global memory size:     " << device_param<cl_ulong>(device,
		CL_DEVICE_GLOBAL_MEM_SIZE) / mb << " MBs" << std::endl;
	std::cout << "Local memory size:      " << device_param<cl_ulong>(device,
		CL_DEVICE_LOCAL_MEM_SIZE) / kb << " KBs" << std::endl;
	std::cout << "Clock frequency:        " << device_param<cl_uint>(device,
		CL_DEVICE_MAX_CLOCK_FREQUENCY) << " MHz" << std::endl;
	std::cout << "Compute units:          " << device_param<cl_uint>(device,
		CL_DEVICE_MAX_COMPUTE_UNITS) << " units" << std::endl;
	std::cout << "Max memory object:      " << device_param<cl_ulong>(device,
		CL_DEVICE_MAX_MEM_ALLOC_SIZE) / mb << " MBs" << std::endl;
	std::cout << "Max work group size:    " << device_param<size_t>(device,
		CL_DEVICE_MAX_WORK_GROUP_SIZE) << " items" << std::endl;
}

hardware::~hardware() { clReleaseDevice(cur_device); }