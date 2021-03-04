#include"im_executors.h"

#define KERN_LANCZOS 1
#define KERN_SPLINE -1
#define KERN_BILINEAR 2
#define KERN_PRECISE -2

#define MITCHELL 0
#define CATMULL 1
#define ADOBE 2
#define B_SPLINE 3

zoomer::zoomer(cl_context context, cl_command_queue queue, 
	functions* conv_kernels) : executor(context, queue, conv_kernels) {
	/* Preallocate BC polynomials */
	polynomials[MITCHELL] = calc_spline_polynom(1 / 3.0f, 1 / 3.0f);
	polynomials[CATMULL] = calc_spline_polynom(0.0f, 0.5f);
	polynomials[ADOBE] = calc_spline_polynom(0.0f, 0.75f);
	polynomials[B_SPLINE] = calc_spline_polynom(1.0f, 0.5f);
}

cl_mem zoomer::calc_spline_polynom(float B, float C) {
	cl_int ret_code;
	float polynom[8] = {
		6.0f - 2.0f * B,                8.0f * B + 24.0f * C,
		0.0f,                          -12.0f * B - 48.0f + C,
		-18.0f + 12.0f * B + 6.0f * C,  6.0f * B + 30.0f * C,
		12.0f - 9.0f * B - 6.0f * C,   -1.0f * B - 6.0f * C
	};
	cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) * 8, polynom, &ret_code);
	util::assert_success(ret_code, "Failed to allocate buffer");
	return buffer;
}

zoomer::~zoomer() {
	for (size_t filter = 0; filter < 4; ++filter) 
	{ clReleaseMemObject(polynomials[filter]); }
}

im_ptr zoomer::run(const std::string& kernel_type, float factor, im_object& src) {
	int func = 0, order = 1; cl_mem polynom = nullptr;
	cl_kernel kern; bool default_sampler = true;

	/* Switch kernel and its params */
	if (kernel_type == "lan3") { func = KERN_LANCZOS; order = 1; }
	else if (kernel_type == "lan4") { func = KERN_LANCZOS; order = 2; }
	else if (kernel_type == "lan5") { func = KERN_LANCZOS; order = 3; }
	if (func == KERN_LANCZOS) { kern = kernels->at("lanczos"); goto end_switch; }
	else if (kernel_type == "mitchell") { func = KERN_SPLINE; polynom = polynomials[MITCHELL]; }
	else if (kernel_type == "catmull") { func = KERN_SPLINE; polynom = polynomials[CATMULL]; }
	else if (kernel_type == "adobe") { func = KERN_SPLINE; polynom = polynomials[ADOBE]; }
	else if (kernel_type == "b-spline") { func = KERN_SPLINE; polynom = polynomials[B_SPLINE]; }
	if (func == KERN_SPLINE) { kern = kernels->at("splines"); goto end_switch; }
	else if (kernel_type == "precise") { func = KERN_PRECISE; kern = kernels->at("precise"); }
	else if (kernel_type == "bilinear") { func = KERN_BILINEAR; 
		kern = kernels->at("bilinear"); default_sampler = false; }
	else  { throw std::runtime_error("Unknown kernel type " + kernel_type); }

	end_switch:
	bool upscale = factor >= 1.0f, first = true;
	float dyn_factor = factor;
	im_object* from = &src; im_object* to = nullptr;

	/* Zooming no more, than twice at one step, get linear spaced object */
	while ((upscale && dyn_factor > 2.0f) || (!upscale && dyn_factor < 0.5f)) {
		int new_width = static_cast<int>(from->size.x * dyn_factor);
		int new_height = static_cast<int>(from->size.y * dyn_factor);
		to = new im_object({ new_width, new_height }, context, queue, NO_BUFFER_WRITE, default_sampler);
		set_common_args(kern, *from, *to, NO_BUFFER_WRITE);
		set_extra_args(kern, *from, *to, func, dyn_factor, order, polynom);
		run_blocking(kern, to->size); std::swap(from, to);
		if (!first) { delete to; } else { first = false; }
		dyn_factor = upscale ? dyn_factor / 2.0f : dyn_factor * 2.0f;
	}

	/* Last step, get gamma corrected image */
	int final_width = static_cast<int>(from->size.x * dyn_factor);
	int final_height = static_cast<int>(from->size.y * dyn_factor);
	im_ptr last = std::make_shared<im_object>(cl_int2{ final_width,
		final_height },context, queue, WRITE_TO_BUFFER, default_sampler);
	set_common_args(kern, *from, *last, WRITE_TO_BUFFER);
	set_extra_args(kern, *from, *last, func, dyn_factor, order, polynom);
	run_blocking(kern, last->size);
	if (!first) { delete from; }
	return std::move(last);
}

/* Set args depending on kernel type */
void zoomer::set_extra_args(cl_kernel kern, im_object& src, 
	im_object& dst, int func, float factor, int order, cl_mem polynom) {
	cl_float2 fact = { factor, factor }; cl_int ret_code = 0;
	switch (func) {
	case KERN_LANCZOS:
		ret_code = clSetKernelArg(kern, 6, sizeof(cl_float2), &fact);
		ret_code |= clSetKernelArg(kern, 7, sizeof(int), &order);
		break;
	case KERN_BILINEAR:
		ret_code = clSetKernelArg(kern, 6, sizeof(cl_float2), &fact);
		break;
	case KERN_SPLINE:
		ret_code = clSetKernelArg(kern, 6, sizeof(cl_float2), &fact);
		ret_code |= clSetKernelArg(kern, 7, sizeof(cl_mem), &polynom);
		break;
	case KERN_PRECISE:
		ret_code = clSetKernelArg(kern, 6, sizeof(cl_int2), &src.size);
		int gcd_w = util::euclidean_gcd(src.size.x, dst.size.x);
		int gcd_h = util::euclidean_gcd(src.size.y, dst.size.y);
		cl_int2 split_out = { src.size.x / gcd_w, src.size.y / gcd_h };
		ret_code |= clSetKernelArg(kern, 7, sizeof(cl_int2), &split_out);
		cl_int2 split_in = { src.size.y / gcd_w, dst.size.y / gcd_h };
		ret_code |= clSetKernelArg(kern, 8, sizeof(cl_int2), &split_in);
		double area = 1.0 / ((double)split_out.x * (double)split_out.y);
		ret_code |= clSetKernelArg(kern, 9, sizeof(double), &area);
	}
	util::assert_success(ret_code, "Failed to set extra args");
}