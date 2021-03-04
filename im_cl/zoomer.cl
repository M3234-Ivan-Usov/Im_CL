
#define BOARD 0.0031308f
#define POWER (float4)(1.0f / 2.4f)

#define IS_LINEAR(p) (float4)(p.x < BOARD, p.y < BOARD, p.z < BOARD, 0.0f)
#define TO_UINT(p) (uint4)(rint(p.x), rint(p.y), rint(p.z), 0)
#define G(p, n_lin, lin) (n_lin * (1.055f * pow(p, POWER) - 0.055f) + lin * 12.92f * p)

#define LAN_ORDER_MAX 3
#define LAN_ARR_MAX 7
#define SPLINE_ORDER 1
#define SPLINE_ARR 3

#define WRITE_TO_BUF 1

__kernel void bilinear(__read_only image2d_t src, __global char* buf,
	__write_only image2d_t dst, sampler_t sampler, int2 out_size, int write_mode, float2 factor) {

	int2 out_coord = (int2) (get_global_id(0), get_global_id(1));
	if (out_coord.x < out_size.x && out_coord.y < out_size.y) {
		float2 base_coord = (float2)(out_coord.x, out_coord.y) / factor;
		float4 out_val = read_imagef(src, sampler, base_coord);

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

__kernel void lanczos(__read_only image2d_t src, __global char* buf, __write_only image2d_t dst,
	sampler_t sampler, int2 out_size, int write_mode, float2 factor, int order) {

	int2 out_coord = (int2) (get_global_id(0), get_global_id(1));
	if (out_coord.x < out_size.x && out_coord.y < out_size.y) {
		float2 base_coord_f;
		float2 frac = modf((float2)(out_coord.x, out_coord.y) / factor, &base_coord_f);
		int2 base_coord = (int2)(rint(base_coord_f.x), rint(base_coord_f.y));

		float2 order_f = (float2)(order), pi = (float2)(M_PI);
		float4 output = (float4)(0.0f), kern_sum = (float4)(0.0f);
		float x_vals[LAN_ARR_MAX], y_vals[LAN_ARR_MAX];

		for (int off = -order; off <= order; ++off) {
			float2 pi_val = ((float2)(off) - frac) * pi;
			x_vals[off + LAN_ORDER_MAX] = (pi_val.x == 0.0f)? 1.0f : 
				sin(pi_val.x) / (pi_val.x) * sin(pi_val.x / order_f.x) / (pi_val.x / order_f.x);
			y_vals[off + LAN_ORDER_MAX] = (pi_val.y == 0.0f) ? 1.0f :
				sin(pi_val.y) / (pi_val.y) * sin(pi_val.y / order_f.y) / (pi_val.y / order_f.y);
		}

		for (int y = -order; y <= order; ++y) {
			for (int x = -order; x <= order; ++x) {
				float cur_weight = x_vals[x + LAN_ORDER_MAX] * y_vals[y + LAN_ORDER_MAX];
				kern_sum += cur_weight;
				float4 in_val = read_imagef(src, sampler, (int2)(base_coord.x + x, base_coord.y + y));
				output += in_val * cur_weight;
			}
		}
		output /= kern_sum;

		write_imagef(dst, out_coord, output);
		if (write_mode == WRITE_TO_BUF) {
			float4 linear = IS_LINEAR(output);
			float4 non_linear = (float4)(1.0f) - linear;
			output = G(output, non_linear, linear) * 255.0f;
			int index = 3 * (out_coord.y * out_size.x + out_coord.x);
			buf[index + 0] = (char)rint(output.x);
			buf[index + 1] = (char)rint(output.y);
			buf[index + 2] = (char)rint(output.z);
		}
	}
}

__kernel void splines(__read_only image2d_t src, __global char* buf, __write_only image2d_t dst,
	sampler_t sampler, int2 out_size, int write_mode, float2 factor, __constant float* polynom) {

	int2 out_coord = (int2) (get_global_id(0), get_global_id(1));
	if (out_coord.x < out_size.x && out_coord.y < out_size.y) {
		float2 base_coord_f; 
		float2 frac = modf((float2)(out_coord.x, out_coord.y) / factor, &base_coord_f);
		int2 base_coord = (int2)(base_coord_f.x, base_coord_f.y);
		float4 output = (float4)(0.0f), kern_sum = (float4)(0.0f);
		float x_vals[SPLINE_ARR], y_vals[SPLINE_ARR];

		for (int off = -SPLINE_ORDER; off <= SPLINE_ORDER; ++off) {
			float2 val = fabs((float2)(off) - frac);
			float2 multiplicator = (float2)(1.0f), cur_pix = (float2)(0.0f);
			int x_p = (val.x < 1.0f) ? 0 : 1, y_p = (val.y < 1.0f) ? 0 : 1;
			for (; x_p < 8 && y_p < 8; x_p += 2, y_p += 2) {
				cur_pix.x += polynom[x_p] * multiplicator.x;
				cur_pix.y += polynom[y_p] * multiplicator.y;
				multiplicator *= val;
			}
			x_vals[off + SPLINE_ORDER] = cur_pix.x / 6.0f;
			y_vals[off + SPLINE_ORDER] = cur_pix.y / 6.0f;
		}

		for (int y = -SPLINE_ORDER; y <= SPLINE_ORDER; ++y) {
			for (int x = -SPLINE_ORDER; x <= SPLINE_ORDER; ++x) {
				float cur_weight = x_vals[x + SPLINE_ORDER] * y_vals[y + SPLINE_ORDER];
				float4 in_val = read_imagef(src, sampler, (int2)(base_coord.x + x, base_coord.y + y));
				kern_sum += cur_weight;
				output += in_val * cur_weight;
			}
		}
		output /= kern_sum;

		write_imagef(dst, out_coord, output);
		if (write_mode == WRITE_TO_BUF) {
			float4 linear = IS_LINEAR(output);
			float4 non_linear = (float4)(1.0f) - linear;
			output = G(output, non_linear, linear) * 255.0f;
			int index = 3 * (out_coord.y * out_size.x + out_coord.x);
			buf[index + 0] = (char)rint(output.x);
			buf[index + 1] = (char)rint(output.y);
			buf[index + 2] = (char)rint(output.z);
		}
	}
}


__kernel void precise(__read_only image2d_t src, __global char* buf, __write_only image2d_t dst,
	sampler_t sampler, int2 out_size, int write_mode, int2 in_size, int2 split_out, int2 split_in, double area) {

	int2 out_coord = (int2) (get_global_id(0), get_global_id(1));
	if (out_coord.x < out_size.x && out_coord.y < out_size.y) {
		int2 cur_pix = (out_coord * split_out) / split_in;
		int2 in_pix = (out_coord * split_out) % split_in;
		int2 start_pix = cur_pix;
		float4 cur_val = read_imagef(src, sampler, cur_pix);
		double4 cur_val_d = (double4)(cur_val.x, cur_val.y, cur_val.z, 0.0);
		double4 output = (double4)(0.0);

		for (int y = 0; y < split_out.y; ++y, in_pix.y++) {
			if (in_pix.y == split_in.y) {
				in_pix.y = 0; cur_pix.y++;
				cur_val = read_imagef(src, sampler, cur_pix);
				cur_val_d = (double4)(cur_val.x, cur_val.y, cur_val.z, 0.0);
			}
			for (int x = 0; x < split_out.x; ++x, in_pix.x++) {
				if (in_pix.x == split_in.x) {
					in_pix.x = 0; cur_pix.x++;
					cur_val = read_imagef(src, sampler, cur_pix);
					cur_val_d = (double4)(cur_val.x, cur_val.y, cur_val.z, 0.0);
				}
				output += cur_val_d * area;
			}
			cur_pix.x = start_pix.x;
		}
		float4 out_val = (float4)(output.x, output.y, output.z, 0.0f);

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
