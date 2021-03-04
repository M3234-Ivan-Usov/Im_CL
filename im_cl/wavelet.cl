
#define SIGN(x) ((x > 0.0f)? 1.0f : -1.0f)

#define WAVELET_HORIZONTAL 0
#define WAVELET_VERTICAL 1
#define sqrt_2 (float4)(1.41421356237f);


__kernel void direct_haar(__read_only image2d_t src,
	__write_only image2d_t dst, sampler_t sampler, int2 cur_size, int dim) {

	int2 out_coord = (int2) (get_global_id(0), get_global_id(1));
	if (out_coord.x < cur_size.x && out_coord.y < cur_size.y) {

		int2 pair_coord_1, pair_coord_2, out_coord_2;
		if (dim == WAVELET_HORIZONTAL) {
			pair_coord_1 = (int2)(2 * out_coord.x, out_coord.y);
			pair_coord_2 = (int2)(pair_coord_1.x + 1, pair_coord_1.y);
			out_coord_2 = (int2)(out_coord.x + cur_size.x, out_coord.y);
		}
		else {
			pair_coord_1 = (int2)(out_coord.x, 2 * out_coord.y);
			pair_coord_2 = (int2)(pair_coord_1.x, pair_coord_1.y + 1);
			out_coord_2 = (int2)(out_coord.x, out_coord.y + cur_size.y);
		}

		float4 first_val = read_imagef(src, sampler, pair_coord_1);
		float4 second_val = read_imagef(src, sampler, pair_coord_2);

		float4 sum_half = (first_val + second_val) / sqrt_2;
		float4 diff_half = (first_val - second_val) / sqrt_2;

		write_imagef(dst, out_coord, sum_half);
		write_imagef(dst, out_coord_2, diff_half);
	}
}

__kernel void soft_threshold(__read_only image2d_t src, __global char* buf,
	__write_only image2d_t dst, sampler_t sampler, int2 out_size, int with_gamma, float threshold) {

	int2 out_coord = (int2) (get_global_id(0), get_global_id(1));
	if (out_coord.x < out_size.x && out_coord.y < out_size.y) {
		float4 in_val = read_imagef(src, sampler, out_coord);
		float4 sign = (float4)(SIGN(in_val.x), SIGN(in_val.y), SIGN(in_val.z), 0.0f);
		float4 out_val = sign * fmax((float4)(0.0f), sign * in_val - (float4)(threshold));
		write_imagef(dst, out_coord, out_val);
	}
}

__kernel void inverse_haar(__read_only image2d_t src,
	__write_only image2d_t dst, sampler_t sampler, int2 cur_size, int dim) {

	int2 in_coord = (int2) (get_global_id(0), get_global_id(1));
	if (in_coord.x < cur_size.x && in_coord.y < cur_size.y) {

		int2 pair_coord_1, pair_coord_2, in_coord_2;
		if (dim == WAVELET_HORIZONTAL) {
			pair_coord_1 = (int2)(2 * in_coord.x, in_coord.y);
			pair_coord_2 = (int2)(pair_coord_1.x + 1, pair_coord_1.y);
			in_coord_2 = (int2)(in_coord.x + cur_size.x, in_coord.y);
		}
		else {
			pair_coord_1 = (int2)(in_coord.x, 2 * in_coord.y);
			pair_coord_2 = (int2)(pair_coord_1.x, pair_coord_1.y + 1);
			in_coord_2 = (int2)(in_coord.x, in_coord.y + cur_size.y);
		}

		float4 first_val = read_imagef(src, sampler, in_coord);
		float4 second_val = read_imagef(src, sampler, in_coord_2);

		float4 sum_half = (first_val + second_val) / sqrt_2;
		float4 diff_half = (first_val - second_val) / sqrt_2;

		write_imagef(dst, pair_coord_1, sum_half);
		write_imagef(dst, pair_coord_2, diff_half);
	}
}