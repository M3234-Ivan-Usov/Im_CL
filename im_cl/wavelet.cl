
#define SIGN(x) ((x > 0.0f)? 1.0f : -1.0f)

#define WAVELET_FORWARD 0
#define WAVELET_INVERSE 1
#define sqrt_2 (float4)(1.41421356237f);


__kernel void horizontal_haar(__read_only image2d_t src, sampler_t sampler,
	__write_only image2d_t dst, int2 cur_sz, int direction) {

	int in_x[2], out_x[2];
	int y = get_global_id(1);

	if (direction == WAVELET_FORWARD) {
		out_x[0] = get_global_id(0);
		out_x[1] = get_global_id(0) + cur_sz.x;
		in_x[0] = 2 * out_x[0];
		in_x[1] = 2 * out_x[0] + 1;
	}
	else {
		in_x[0] = get_global_id(0);
		in_x[1] = get_global_id(0) + cur_sz.x;
		out_x[0] = 2 * in_x[0];
		out_x[1] = 2 * in_x[0] + 1;
	}

	float4 first_val = read_imagef(src, sampler, (int2)(in_x[0], y));
	float4 second_val = read_imagef(src, sampler, (int2)(in_x[1], y));
	write_imagef(dst, (int2)(out_x[0], y), (first_val + second_val) / sqrt_2);
	write_imagef(dst, (int2)(out_x[1], y), (first_val - second_val) / sqrt_2);
}

__kernel void vertical_haar(__read_only image2d_t src, sampler_t sampler,
	__write_only image2d_t dst, int2 cur_sz, int direction) {

	int in_y[2], out_y[2];
	int x = get_global_id(0);

	if (direction == WAVELET_FORWARD) {
		out_y[0] = get_global_id(1);
		out_y[1] = get_global_id(1) + cur_sz.y;
		in_y[0] = 2 * out_y[0];
		in_y[1] = 2 * out_y[0] + 1;
	}
	else {
		in_y[0] = get_global_id(1);
		in_y[1] = get_global_id(1) + cur_sz.y;
		out_y[0] = 2 * in_y[0];
		out_y[1] = 2 * in_y[0] + 1;
	}

	float4 first_val = read_imagef(src, sampler, (int2)(x, in_y[0]));
	float4 second_val = read_imagef(src, sampler, (int2)(x, in_y[1]));
	write_imagef(dst, (int2)(x, out_y[0]), (first_val + second_val) / sqrt_2);
	write_imagef(dst, (int2)(x, out_y[1]), (first_val - second_val) / sqrt_2);
}


__kernel void soft_threshold(__read_only image2d_t src, sampler_t sampler,
	__write_only image2d_t dst, float threshold) {

	int2 cd = (int2)(get_global_id(0), get_global_id(1));
	float4 in_val = read_imagef(src, sampler, cd);
	float4 sign = (float4)(SIGN(in_val.x), SIGN(in_val.y), SIGN(in_val.z), SIGN(in_val.w));
	write_imagef(dst, cd, sign * fmax((float4)(0.0f), sign * in_val - (float4)(threshold)));
}