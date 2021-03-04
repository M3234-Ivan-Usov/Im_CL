
#define BOARD 0.0031308f
#define POWER (float4)(1.0f / 2.4f)

#define IS_LINEAR(p) (float4)(p.x < BOARD, p.y < BOARD, p.z < BOARD, 0.0f)
#define TO_UINT(p) (uint4)(rint(p.x), rint(p.y), rint(p.z), 0)
#define G(p, n_lin, lin) (n_lin * (1.055f * pow(p, POWER) - 0.055f) + lin * 12.92f * p)

#define WRITE_TO_BUF 1


__kernel void convolution_2D(__read_only image2d_t src, __global char* buf, __write_only image2d_t dst,
	sampler_t sampler, int2 out_size, int write_mode, int radius, int lin_size, __constant float* kern) {

	int2 out_coord = (int2) (get_global_id(0), get_global_id(1));
	if (out_coord.x < out_size.x && out_coord.y < out_size.y) {
		float4 out_val = (float4)(0.0f);
		for (int y = 0; y < lin_size; ++y) {
			for (int x = 0; x < lin_size; ++x) {
				int2 in_coord = (int2)(out_coord.x + x - radius, out_coord.y + y - radius);
				float4 in_val = read_imagef(src, sampler, in_coord);
				out_val += in_val * kern[y * lin_size + x];
			}
		}

		write_imagef(dst, out_coord, out_val);
		if (write_mode == WRITE_TO_BUF) {
			float4 linear = IS_LINEAR(out_val);
			float4 non_linear = (float4)(1.0f) - linear;
			out_val = G(out_val, non_linear, linear) * 255.0f;
			int index = 3 * (out_coord.y * out_size.x + out_coord.x);
			buf[index + 0] = (char)rint(out_val.x);
			buf[index + 1] = (char)rint(out_val.y);
			buf[index + 2] = (char)rint(out_val.z);
		}
	}
}

__kernel void horizontal_conv(__read_only image2d_t src, __global char* buf, __write_only image2d_t dst,
	sampler_t sampler, int2 out_size, int write_mode, int radius, __constant float* kern) {

	int2 out_coord = (int2) (get_global_id(0), get_global_id(1));
	if (out_coord.x < out_size.x && out_coord.y < out_size.y) {
		float4 out_val = (float4)(0.0f);
		for (int x = -radius; x <= radius; ++x) {
			float4 in_val = read_imagef(src, sampler, (int2)(out_coord.x + x, out_coord.y));
			out_val += in_val * kern[radius + x];
		}

		write_imagef(dst, out_coord, out_val);
		if (write_mode == WRITE_TO_BUF) {
			float4 linear = IS_LINEAR(out_val);
			float4 non_linear = (float4)(1.0f) - linear;
			out_val = G(out_val, non_linear, linear) * 255.0f;
			int index = 3 * (out_coord.y * out_size.x + out_coord.x);
			buf[index + 0] = (char)rint(out_val.x);
			buf[index + 1] = (char)rint(out_val.y);
			buf[index + 2] = (char)rint(out_val.z);
		}
	}
}

__kernel void vertical_conv(__read_only image2d_t src, __global char* buf, __write_only image2d_t dst,
	sampler_t sampler, int2 out_size, int write_mode, int radius, __constant float* kern) {

	int2 out_coord = (int2) (get_global_id(0), get_global_id(1));
	if (out_coord.x < out_size.x && out_coord.y < out_size.y) {
		float4 out_val = (float4)(0.0f);
		for (int y = -radius; y <= radius; ++y) {
			float4 in_val = read_imagef(src, sampler, (int2)(out_coord.x, out_coord.y + y));
			out_val += in_val * kern[radius + y];
		}

		write_imagef(dst, out_coord, out_val);
		if (write_mode == WRITE_TO_BUF) {
			float4 linear = IS_LINEAR(out_val);
			float4 non_linear = (float4)(1.0f) - linear;
			out_val = G(out_val, non_linear, linear) * 255.0f;
			int index = 3 * (out_coord.y * out_size.x + out_coord.x);
			buf[index + 0] = (char)rint(out_val.x);
			buf[index + 1] = (char)rint(out_val.y);
			buf[index + 2] = (char)rint(out_val.z);
		}
	}
}