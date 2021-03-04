#include"im_executors.h"

rotator::rotator(cl_context context, cl_command_queue queue,
	functions* kernels) : executor(context, queue, kernels) {}

/* Estimate rotated image new size */
cl_int2 rotator::rotate_size(const im_object& src, double theta) {
	double cos_t = cos(theta), sin_t = sin(theta);
	cl_int r_w = static_cast<cl_int>(fabs(src.size.x * cos_t) + fabs(src.size.y * sin_t)) + 1;
	cl_int r_h = static_cast<cl_int>(fabs(src.size.y * cos_t) + fabs(src.size.x * sin_t)) + 1;
	return { r_w, r_h };
}

/* Calculate image center */
cl_int2 rotator::im_center(const im_object& src) {
	cl_int c_x = (src.size.x + 1) / 2 - 1;
	cl_int c_y = (src.size.y + 1) / 2 - 1;
	return { c_x, c_y };
};

void rotator::set_coord_args(cl_kernel kern, const cl_int2& in_size, 
	const cl_int2& src_center, const cl_int2& dst_center) {
	cl_int ret_code = clSetKernelArg(kern, 5, sizeof(cl_int2), &in_size);
	ret_code |= clSetKernelArg(kern, 6, sizeof(cl_int2), &src_center);
	ret_code |= clSetKernelArg(kern, 7, sizeof(cl_int2), &dst_center);
	util::assert_success(ret_code, "Failed to set coordinates");
}

im_ptr rotator::run(const std::string& algo, double theta, im_object& src) {
	double th_trunc; cl_kernel kern; cl_int ret_code = 0;
	if (modf(theta / 90.0, &th_trunc) == 0.0) {
		return simple_angle(static_cast<int>(theta), src);
	}
	double rad_theta = theta / 180.0 * CL_M_PI;
	auto rot_size = rotate_size(src, rad_theta);
	if (algo == "shear") { 
		im_ptr dst = std::make_shared<im_object>(rot_size, context, queue, WRITE_TO_BUFFER, true);
		kern = kernels->at("shear_rotate"); 
		set_common_args(kern, src, *dst, WRITE_TO_BUFFER);
		set_coord_args(kern, src.size, im_center(src), im_center(*dst));
		double shear_alpha = -tan(rad_theta / 2.0), shear_beta = sin(rad_theta);
		ret_code = clSetKernelArg(kern, 8, sizeof(double), &shear_alpha);
		ret_code |= clSetKernelArg(kern, 9, sizeof(double), &shear_beta);
		util::assert_success(ret_code, "Failed to set shear angles");
		run_blocking(kern, src.size);
		return std::move(dst);
	}
	else if (algo == "map") {
		im_ptr dst = std::make_shared<im_object>(rot_size, context, queue, WRITE_TO_BUFFER, false);
		kern = kernels->at("map_rotate");
		set_common_args(kern, src, *dst, WRITE_TO_BUFFER);
		set_coord_args(kern, src.size, im_center(src), im_center(*dst));
		double sin_t = sin(rad_theta), cos_t = cos(rad_theta);
		ret_code |= clSetKernelArg(kern, 8, sizeof(double), &sin_t);
		ret_code |= clSetKernelArg(kern, 9, sizeof(double), &cos_t);
		util::assert_success(ret_code, "Failed to set map angles");
		run_blocking(kern, src.size);
		return std::move(dst);
	}
	else { throw std::runtime_error("Unknown rotation " + algo); }
}

im_ptr rotator::simple_angle(int theta, im_object& src) {
	cl_kernel kern; cl_int2 dst_size = src.size;
	if (theta == 90) { kern = kernels->at("clockwise"); std::swap(dst_size.x, dst_size.y); }
	else if (theta == -90) { kern = kernels->at("counter_clockwise");  std::swap(dst_size.x, dst_size.y); }
	else if (theta == 180 || theta == -180) { kern = kernels->at("flip"); }
	else { throw std::runtime_error("Not a simple angle"); }
	im_ptr dst = std::make_shared<im_object>(dst_size, context, queue, WRITE_TO_BUFFER);
	set_common_args(kern, src, *dst, WRITE_TO_BUFFER);
	run_blocking(kern, dst_size);
	return std::move(dst);
}