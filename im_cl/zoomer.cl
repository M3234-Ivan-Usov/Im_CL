
/* 
*	Bilinear Interpolation
*   Using CL_FILTER_BILINEAR sampler address mode
*/
__kernel void bilinear(__read_only image2d_t src,
	sampler_t sampler, __write_only image2d_t dst, float2 factor) {

	int2 out_cd = (int2)(get_global_id(0), get_global_id(1));
	float2 base_cd = (float2)(out_cd.x / factor.x, out_cd.y / factor.y);
	float4 out_val = read_imagef(src, sampler, base_cd);
	write_imagef(dst, out_cd, out_val);
}


/* 
*	Lanczos. Using sinc function
*/

float sinc(float val, int order) {
	float pi_val = val * M_PI;
	float order_val = pi_val / (float)order;
	return (pi_val == 0.0f) ? 1.0f : (native_sin(pi_val) / pi_val) * (native_sin(order_val) / order_val);
}

__kernel void lanczos(__read_only image2d_t src, sampler_t sampler,
	__write_only image2d_t dst, float2 factor, int order) {

	int2 out_cd = (int2)(get_global_id(0), get_global_id(1));
	float2 base_cd_f;
	float2 frac = modf(convert_float2(out_cd) / factor, &base_cd_f);
	int2 base_cd = convert_int2(rint(base_cd_f));
	
	float kern_sum = 0.0f;
	float4 output = 0.0f;
	for (int y = -order; y <= order; ++y) {
		for (int x = -order; x <= order; ++x) {
			float2 off_val = ((float2)(x, y) - frac);
			int2 cur_cd = (int2)(base_cd.x + x, base_cd.y + y);
			float4 cur_val = read_imagef(src, sampler, cur_cd);
			float cur_weight = sinc(off_val.x, order) * sinc(off_val.y, order);
			kern_sum += cur_weight;
			output += cur_val * cur_weight;
		}
	}
	output = fmax((float4)0.0f, fmin(1.0f, output / kern_sum));
	write_imagef(dst, out_cd, output);
}


/*
*	BC-Splines. Using polynomials
*/

#define SPLINE_ORDER 2

float spl(float val, __constant float* upper, __constant float* lower) {
	float v = fabs(val);
	float v_sqr = v * v;
	float v_cb = v_sqr * v;
	if (v < 1.0f) { return (upper[0] + v * upper[1] + v_sqr * upper[2] + v_cb * upper[3]) / 6.0f; }
	else if (v < 2.0f) { return (lower[0] + v * lower[1] + v_sqr * lower[2] + v_cb * lower[3]) / 6.0f; }
	else { return 0.0f; }
}

__kernel void splines(__read_only image2d_t src, sampler_t sampler, __write_only image2d_t dst,
	float2 factor, __constant float* upper, __constant float* lower) {

	int2 out_cd = (int2)(get_global_id(0), get_global_id(1));
	float2 base_cd_f;
	float2 frac = modf(convert_float2(out_cd) / factor, &base_cd_f);
	int2 base_cd = convert_int2(rint(base_cd_f));

	float kern_sum = 0.0f;
	float4 output = 0.0f;
	for (int y = -SPLINE_ORDER; y <= SPLINE_ORDER; ++y) {
		for (int x = -SPLINE_ORDER; x <= SPLINE_ORDER; ++x) {
			float2 off_val = ((float2)(x, y) - frac);
			int2 cur_cd = (int2)(base_cd.x + x, base_cd.y + y);
			float4 cur_val = read_imagef(src, sampler, cur_cd);
			float cur_weight = spl(off_val.x, upper, lower) * spl(off_val.y, upper, lower);
			kern_sum += cur_weight;
			output += cur_val * cur_weight;
		}
	}
	output = fmax((float4)0.0f, fmin(1.0f, output / kern_sum));
	write_imagef(dst, out_cd, output);
}


/* 
*	Precise method
*/
__kernel void precise(__read_only image2d_t src, sampler_t sampler,
	__write_only image2d_t dst, int2 split_out, int2 split_in, float area) {

	int2 out_cd = (int2)(get_global_id(0), get_global_id(1));
	int2 cur_pix = (out_cd * split_out) / split_in;
	int2 in_pix = (out_cd * split_out) % split_in;
	int2 start_pix = cur_pix;
	float4 cur_val, out_val = 0.0f;

	for (int y = 0; y < split_out.y; ++y, in_pix.y++) {
		if (in_pix.y == split_in.y) {
			in_pix.y = 0; cur_pix.y++;
			cur_val = read_imagef(src, sampler, cur_pix);
		}
		for (int x = 0; x < split_out.x; ++x, in_pix.x++) {
			if (in_pix.x == split_in.x) {
				in_pix.x = 0; cur_pix.x++;
				cur_val = read_imagef(src, sampler, cur_pix);
			}
			out_val += cur_val * area;
		}
		cur_pix.x = start_pix.x;
	}
	write_imagef(dst, out_cd, out_val);
}