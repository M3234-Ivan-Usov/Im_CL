#include"im_object.h"
#include"util.h"

float im_object::gamma_map[];
float im_object::norm_map[];
cl_mem im_object::empty_buffer = nullptr;

im_object::im_object(cl_int2 size, cl_context context, cl_command_queue queue,
	int write_mode, bool def_sampler) : size(size), queue(queue) {

	this->init_image_stuff();
	this->create_mem(nullptr, context, CL_MEM_READ_WRITE, def_sampler);
	if (write_mode == WRITE_TO_BUFFER) {
		size_t buf_size = 3ul * size.x * size.y; cl_int ret_code;
		cl_storage = clCreateBuffer(context, CL_MEM_READ_WRITE, buf_size, nullptr, &ret_code);
		util::assert_success(ret_code, "Failed to allocate buffer");
		is_empty_buffer = false;
	}
}

im_object::im_object(char* host_ptr, size_t width, size_t height, cl_context context,
	cl_command_queue queue, bool convert_to_linear, bool def_sampler): host_ptr(host_ptr), queue(queue) {

	size = { (cl_int)width, (cl_int)height }; this->init_image_stuff();
	float* ext_channels = extend_channels(reinterpret_cast<byte*>(host_ptr),
		convert_to_linear? gamma_map : norm_map, width * height);
	this->create_mem(ext_channels, context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, def_sampler);
	delete[] ext_channels;
}

im_object::im_object(im_object&& other) : cl_im(other.cl_im), cl_storage(other.cl_storage), queue(other.queue),
	sampler(other.sampler), size(other.size), descriptor(other.descriptor), format(other.format),
	host_ptr(other.host_ptr), values(other.values), is_empty_buffer(other.is_empty_buffer) {

	if (!is_empty_buffer) { clRetainMemObject(cl_storage); } 
	clRetainMemObject(cl_im); clRetainSampler(sampler);
	other.host_ptr = nullptr; other.values = nullptr;
}

void im_object::init_image_stuff() {
	descriptor.image_width = static_cast<size_t>(size.x);
	descriptor.image_height = static_cast<size_t>(size.y);

	format.image_channel_data_type = CL_FLOAT;
	format.image_channel_order = CL_RGBA;
	descriptor.image_type = CL_MEM_OBJECT_IMAGE2D;

	descriptor.image_row_pitch = 0; descriptor.image_slice_pitch = 0;
	descriptor.num_mip_levels = 0; descriptor.num_samples = 0;
	descriptor.image_depth = 1; descriptor.image_array_size = 1;
}

void im_object::create_mem(float* rgba_ptr, cl_context context, cl_mem_flags flags, bool def_sampler) {
	cl_int ret_code;
	cl_im = clCreateImage(context, flags, &format, &descriptor, rgba_ptr, &ret_code);
	util::assert_success(ret_code, "Failed to create image object");

	sampler = def_sampler ?
		clCreateSampler(context, CL_FALSE, CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_NEAREST, &ret_code) :
		clCreateSampler(context, CL_FALSE, CL_ADDRESS_MIRRORED_REPEAT, CL_FILTER_LINEAR, &ret_code);
	util::assert_success(ret_code, "Failed to create default sampler");
}

float* im_object::extend_channels(byte* src, float* extension_map, size_t pixels) {
	float* extended = new float[4 * pixels];
	values = new im_values;
	for (size_t pix = 0, src_pix = 0, ext_pix = 0;
		pix < pixels; ++pix, src_pix += 3, ext_pix += 4) {

		for (size_t col = 0; col < 3; ++col) {
			int val = static_cast<int>(src[src_pix + col]);
			float flt_val = norm_map[val];
			values->mean[col] += flt_val; values->mean[3] += flt_val;
			values->histograms[col][val]++; values->histograms[3][val]++;
			extended[ext_pix + col] = extension_map[val];
		}
		extended[ext_pix + 3] = 0.0f;
	}

	for (size_t col = 0; col < 3; ++col) { values->mean[col] /= pixels; }
	values->mean[3] /= 3 * pixels;
	values->cached_hist = true, values->cached_mean = true;
	return extended;
}

char* im_object::get_host_ptr() {
	if (host_ptr == nullptr) {
		size_t mem_amount = 3ul * size.x * size.y;
		host_ptr = new char[mem_amount];
		size_t origin[3] = { 0, 0, 0 };
		size_t region[3] = { (size_t)size.x, (size_t)size.y, 1 };
		cl_int ret_code = clEnqueueReadBuffer(queue, cl_storage,
			CL_TRUE, 0, mem_amount, host_ptr, 0, NULL, NULL);
		util::assert_success(ret_code, "Failed to read from device");
		clReleaseMemObject(cl_storage);
		cl_storage = empty_buffer;
		is_empty_buffer = true;
	}
	return host_ptr;
}

int** im_object::historgrams() {
	if (values == nullptr) { values = new im_values; }
	auto src = reinterpret_cast<byte*>(get_host_ptr());
	size_t pixels = static_cast<size_t>(size.x * size.y);
	return values->calc_historgam(src, pixels);
}

std::pair<float*, float*>im_object::stat() {
	if (values == nullptr) { values = new im_values; }
	auto src = reinterpret_cast<byte*>(get_host_ptr());
	size_t pixels = static_cast<size_t>(size.x * size.y);
	return std::make_pair(values->calc_mean(src, pixels), values->calc_variance(src, pixels));
}


im_object::~im_object() {
	if (!is_empty_buffer) { clReleaseMemObject(cl_storage); }
	clReleaseMemObject(cl_im); clReleaseSampler(sampler);
	delete[] host_ptr; delete values;
}