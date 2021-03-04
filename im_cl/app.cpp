#include"app.h"
#include<fstream>

#define EMPTY_SIZE 1

app::app(size_t plat_id, size_t dev_id) : env(plat_id, dev_id) {
	std::cout << "Initialising...";
	this->match_extensions();
	this->match_kernels();
	this->compile_kernels();
	this->init_executors();
	std::cout << " Ready" << std::endl;
}

app::~app() {
	delete zoomer_ptr;
	delete converser_ptr;
	delete rotator_ptr;
	delete filter_ptr;
	delete wavelet_ptr;
	delete contraster_ptr;

	clReleaseContext(context);
	clReleaseCommandQueue(queue);
	clReleaseMemObject(im_object::empty_buffer);
	for (cl_program program : prog_objects) { clReleaseProgram(program); }
	for (auto prog : prog_tree) {
		for (auto kern : prog.second) { clReleaseKernel(kern.second); }
	}
}

void app::compile_kernels() {
	cl_int ret_code;
	cl_context_properties contextProperties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)env.cur_platform, 0 };
	context = clCreateContext(contextProperties, 1, &env.cur_device, NULL, NULL, &ret_code);
	util::assert_success(ret_code, "Failed to create context");
	for (auto prog_it = prog_tree.begin(); prog_it != prog_tree.end(); ++prog_it) {
		std::ifstream src_file(prog_it->first);
		std::string src_program(std::istreambuf_iterator<char>(src_file), (std::istreambuf_iterator<char>()));
		const char* src = src_program.c_str();
		size_t length = src_program.length();
		prog_objects.push_back(clCreateProgramWithSource(context, 1, &src, &length, &ret_code));
		ret_code = clBuildProgram(prog_objects.back(), 1, &env.cur_device, "-I.", NULL, NULL);
		if (ret_code != CL_SUCCESS) {
			char* log_str = new char[12148]; cl_uint ret_size;
			clGetProgramBuildInfo(prog_objects.back(), env.cur_device, CL_PROGRAM_BUILD_LOG, 12148, log_str, &ret_size);
			std::string build_log(log_str, ret_size); delete[] log_str;
			throw std::runtime_error(build_log);
		}
		for (auto kern_it = prog_it->second.begin(); kern_it != prog_it->second.end(); ++kern_it) {
			kern_it->second = clCreateKernel(prog_objects.back(), kern_it->first.c_str(), &ret_code);
			util::assert_success(ret_code, "Failed to create kernel " + kern_it->first);
		}
	}
	queue = clCreateCommandQueue(context, env.cur_device, 0, &ret_code);
	util::assert_success(ret_code, "Failed to create command queue");
}

void app::match_kernels() {
	prog_tree.emplace("zoomer.cl", util::map_of({"lanczos", "bilinear", "splines", "precise"}));

	prog_tree.emplace("rotator.cl", util::map_of({"clockwise", "counter_clockwise", "flip", "shear_rotate", "map_rotate"}));

	prog_tree.emplace("filter.cl", util::map_of({"horizontal_conv", "vertical_conv", "convolution_2D"}));

	prog_tree.emplace("wavelet.cl", util::map_of({"direct_haar", "inverse_haar", "soft_threshold"}));

	prog_tree.emplace("converser.cl", util::map_of({ "gamma_correction", "srgb_to_ycbcr", "ycbcr_to_srgb",
		"srgb_to_hsv", "hsv_to_srgb", "srgb_to_hsl", "hsl_to_srgb", "hsl_to_hsv", "hsv_to_hsl",
		"srgb_to_ciexyz", "ciexyz_to_srgb", "ciexyz_to_cielab", "cielab_to_ciexyz" }));

	prog_tree.emplace("contraster.cl", util::map_of({ "exclusive_hist", "adaptive_hist_all", "adaptive_hist_single", "manual" }));
}


void app::match_extensions() {
	loader.emplace(".pnm", io_manager::load_pnm);
	writer.emplace(".pnm", io_manager::write_pnm);
}

void app::init_executors() {
	cl_int ret_code; 
	im_object::empty_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, EMPTY_SIZE, nullptr, &ret_code);
	util::assert_success(ret_code, "Failed to create empty buffer");
	zoomer_ptr = new zoomer(context, queue, &prog_tree.at("zoomer.cl"));
	converser_ptr = new converser(context, queue, &prog_tree.at("converser.cl"));
	rotator_ptr = new rotator(context, queue, &prog_tree.at("rotator.cl"));
	filter_ptr = new filter(context, queue, &prog_tree.at("filter.cl"));
	wavelet_ptr = new wavelet(context, queue, &prog_tree.at("wavelet.cl"),
		prog_tree.at("converser.cl").at("gamma_correction"));
	contraster_ptr = new contraster(context, queue, &prog_tree.at("contraster.cl"), converser_ptr);
}


void app::env_info() {
	char* log_str = new char[2048];
	for (size_t pl_id = 0; pl_id < env.plat_num; ++pl_id) {
		cl_uint dev_num, ret_size;
		clGetPlatformInfo(env.platforms[pl_id], CL_PLATFORM_NAME, 2048, log_str, &ret_size);
		std::cout << "  Platform " << pl_id << ": " << std::string(log_str, ret_size) << std::endl;
		clGetDeviceIDs(env.platforms[pl_id], CL_DEVICE_TYPE_ALL, 0, NULL, &dev_num);
		cl_device_id* dev_ids = new cl_device_id[dev_num];
		clGetDeviceIDs(env.platforms[pl_id], CL_DEVICE_TYPE_ALL, dev_num, dev_ids, NULL);
		for (size_t d_id = 0; d_id < dev_num; ++d_id) {
			std::cout << ((env.device_id == d_id && pl_id == env.platform_id) ? "->" : "  ")
				<< "  Device " << d_id << ": " << hardware::string_param(dev_ids[d_id], CL_DEVICE_NAME) <<
				" (" << hardware::string_param(dev_ids[d_id], CL_DEVICE_VERSION) << ")" << std::endl;
			clReleaseDevice(dev_ids[d_id]);
		}
		delete[] dev_ids;
	}
	delete[] log_str;
}

im_ptr app::get_im(std::string filename, bool convert_to_linear) {
	return loader.at(util::file_ext(filename))(context, queue, filename, convert_to_linear);
}

void app::put_im(std::string filename, im_object& im) {
	writer.at(util::file_ext(filename))(im, filename);
}