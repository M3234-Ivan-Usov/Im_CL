#include"im_executors.h"

rotator::rotator(hardware* env, functions* kernels) : executor(env, kernels) {}

cl_int2 rotator::rotate_size(const im_ptr& src, double theta) {
	double cos_t = cos(theta), sin_t = sin(theta);
	cl_int r_w = static_cast<cl_int>(fabs(src->size.x * cos_t) + fabs(src->size.y * sin_t)) + 1;
	cl_int r_h = static_cast<cl_int>(fabs(src->size.y * cos_t) + fabs(src->size.x * sin_t)) + 1;
	return { r_w, r_h };
}

std::pair<cl_int2, cl_int2> rotator::calc_corners(const cl_int2& rot_size,
	const cl_int2& src_size, const cl_int2& center, double theta) {
	cl_int2 padding = { rot_size.x - src_size.x, rot_size.y - src_size.y };
	padding.x /= 2, padding.y /= 2;
	cl_int2 top_left = rotate_point({ padding.x, padding.y }, center, theta);
	cl_int2 bottom_left = rotate_point({ padding.x, rot_size.y - 1 - padding.y }, center, theta);
	cl_int2 top_right = rotate_point({ rot_size.x - 1 - padding.x - 1, padding.y }, center, theta);
	cl_int2 bottom_right = rotate_point({ rot_size.x - 1 - padding.x, rot_size.y - 1 - padding.y }, center, theta);
	cl_int2 start = { std::max(top_left.x, bottom_left.x), std::max(top_left.y, top_right.y) };
	cl_int2 end = { std::min(top_right.x, bottom_right.x), std::min(bottom_left.y, bottom_right.y) };
	start.x += center.x, start.y += center.y;
	end.x += center.x, end.y += center.y;
	return { start, end };
}

cl_int2 rotator::rotate_point(const cl_int2& point, const cl_int2& center, double theta) {
	double cos_t = cos(theta), sin_t = sin(theta);
	cl_int2 coord_point = { point.x - center.x, point.y - center.y };
	return {
		static_cast<cl_int>(coord_point.x * cos_t + coord_point.y * sin_t),
		static_cast<cl_int>(coord_point.y * cos_t - coord_point.x * sin_t)
	};
}


im_ptr rotator::run(const std::string& algo, double theta, const cl_int2& center, im_ptr& src) {
	double rad_theta = theta / 180.0 * CL_M_PI;
	cl_int2 rot_size = rotate_size(src, rad_theta);
	float rad = static_cast<float>(rad_theta);
	cl_float2 angles;
	cl_sampler sampler;
	if (algo == "shear") { 
		angles = { -tanf(rad / 2.0f), sinf(rad) };
		sampler = env->samplers.at({ CL_ADDRESS_CLAMP, CL_FILTER_NEAREST });
	}
	else if (algo == "map") { 
		angles = { sinf(rad), cosf(rad) };
		sampler = env->samplers.at({ CL_ADDRESS_CLAMP, CL_FILTER_LINEAR });
	}
	else { throw std::runtime_error("Unknown rotation: " + algo); }
	cl_kernel kern = kernels->at(algo);
	cl_mem dst = env->alloc_im(rot_size);
	cl_float2 src_center = { (cl_float)center.x, (cl_float)center.y };
	cl_int2 dst_center = {
		static_cast<cl_int>((src_center.x / src->size.x) * rot_size.x),
		static_cast<cl_int>((src_center.y / src->size.y) * rot_size.y)
	};
	cl_int ret_code = set_common_args(kern, src->cl_storage, sampler, dst);
	ret_code |= clSetKernelArg(kern, 3, sizeof(cl_int2), &rot_size);
	ret_code |= clSetKernelArg(kern, 4, sizeof(cl_float2), &src_center);
	ret_code |= clSetKernelArg(kern, 5, sizeof(cl_int2), &dst_center);
	ret_code |= clSetKernelArg(kern, 6, sizeof(cl_float2), &angles);
	cl_event q_event = run_with_event(kern, rot_size);
	auto corners = calc_corners(rot_size, src->size, dst_center, rad_theta);
	cl_int2 reduced_size = { 
		corners.second.x - corners.first.x,
		corners.second.y - corners.first.y 
	};
	im_ptr result = std::make_shared<im_object>(reduced_size, env);
	size_t origin[3] = { (size_t)corners.first.x, (size_t)corners.first.y, 0 };
	size_t zeros[3] = { 0, 0, 0 };
	size_t region[3] = {(size_t) reduced_size.x, (size_t)reduced_size.y, 1 };
	ret_code = clEnqueueCopyImage(env->queue, dst, result->cl_storage, origin,
		zeros, region, 1, &q_event, nullptr);
	ret_code |= clFinish(env->queue);
	clReleaseMemObject(dst);
	return std::move(result);
}

im_ptr rotator::simple_angle(const std::string& direction, im_ptr& src) {
	cl_kernel kern = kern = kernels->at(direction);
	cl_int2 dst_size = { src->size.y, src->size.x };
	cl_sampler sampler = env->samplers.at({ CL_ADDRESS_NONE, CL_FILTER_NEAREST });
	im_ptr dst = std::make_shared<im_object>(dst_size, env);
	cl_int ret_code = set_common_args(kern, src->cl_storage, sampler, dst->cl_storage);
	ret_code |= clSetKernelArg(kern, 3, sizeof(cl_int2), &dst_size);
	run_blocking(kern, dst_size);
	return std::move(dst);
}