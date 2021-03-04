
#define BOARD 0.0031308f
#define POWER (float4)(1.0f / 2.4f)

#define IS_LINEAR(p) (float4)(p.x < BOARD, p.y < BOARD, p.z < BOARD, 0.0f)
#define TO_UINT(p) (uint4)(rint(p.x), rint(p.y), rint(p.z), 0)

__kernel void counter_clockwise(__read_only image2d_t src, 
	__write_only image2d_t dst, sampler_t sampler, int2 out_size, int with_gamma) {
	int2 out_coord = (int2)(get_global_id(0), get_global_id(1));
	if (out_coord.x < out_size.x && out_coord.y < out_size.y) {
		int2 src_coord = (int2)(out_coord.y, out_size.x - out_coord.x - 1);
		float4 out_val = read_imagef(src, sampler, src_coord);
		if (with_gamma == 1) {
			float4 linear = IS_LINEAR(out_val);
			float4 non_linear = (float4)(1.0f) - linear;
			out_val = non_linear * (1.055f * pow(out_val, POWER) - 0.055f) + linear * 12.92f * out_val;
			out_val *= (float4)(255.0f);
			write_imageui(dst, out_coord, TO_UINT(out_val));
		}
		else { write_imagef(dst, out_coord, out_val); }
	}
}

__kernel void clockwise(__read_only image2d_t src,
	__write_only image2d_t dst, sampler_t sampler, int2 out_size, int with_gamma) {
	int2 out_coord = (int2)(get_global_id(0), get_global_id(1));
	if (out_coord.x < out_size.x && out_coord.y < out_size.y) {
		int2 src_coord = (int2)(out_size.y - out_coord.y - 1, out_coord.x);
		float4 out_val = read_imagef(src, sampler, src_coord);
		if (with_gamma == 1) {
			float4 linear = IS_LINEAR(out_val);
			float4 non_linear = (float4)(1.0f) - linear;
			out_val = non_linear * (1.055f * pow(out_val, POWER) - 0.055f) + linear * 12.92f * out_val;
			out_val *= (float4)(255.0f);
			write_imageui(dst, out_coord, TO_UINT(out_val));
		}
		else { write_imagef(dst, out_coord, out_val); }
	}
}

__kernel void flip(__read_only image2d_t src, __write_only image2d_t dst, 
	sampler_t sampler, int2 out_size, int with_gamma) {
	int2 out_coord = (int2)(get_global_id(0), get_global_id(1));
	if (out_coord.x < out_size.x && out_coord.y < out_size.y) {
		int2 src_coord = (int2)(out_size.x - out_coord.x - 1, out_size.y - out_coord.y - 1);
		float4 out_val = read_imagef(src, sampler, src_coord);
		if (with_gamma == 1) {
			float4 linear = IS_LINEAR(out_val);
			float4 non_linear = (float4)(1.0f) - linear;
			out_val = non_linear * (1.055f * pow(out_val, POWER) - 0.055f) + linear * 12.92f * out_val;
			out_val *= (float4)(255.0f);
			write_imageui(dst, out_coord, TO_UINT(out_val));
		}
		else { write_imagef(dst, out_coord, out_val); }
	}
}

__kernel void shear_rotate(__read_only image2d_t src, __write_only image2d_t dst, sampler_t sampler, 
	int2 out_size, int with_gamma, int2 in_size, int2 src_center, int2 dst_center, double alpha, double beta) {
	int2 coord = (int2)(get_global_id(0), get_global_id(1));
	if (coord.x < in_size.x && coord.y < in_size.y) {
		int2 rel_to_center = in_size - coord - src_center - (int2)(1);
		double rot_x = rel_to_center.x + alpha * rel_to_center.y;
		double rot_y = rel_to_center.y + beta * rot_x;
		rot_x += alpha * rot_y;
		int2 rot_coord = (int2)(rint(rot_x), rint(rot_y));
		rot_coord = dst_center - rot_coord;
		if (0 <= rot_coord.x && rot_coord.x < out_size.x &&
			0 <= rot_coord.y && rot_coord.y < out_size.y) {
			float4 out_val = read_imagef(src, sampler, coord);
			if (with_gamma == 1) {
				float4 linear = IS_LINEAR(out_val);
				float4 non_linear = (float4)(1.0f) - linear;
				out_val = non_linear * (1.055f * pow(out_val, POWER) - 0.055f) + linear * 12.92f * out_val;
				out_val *= (float4)(255.0f);
				write_imageui(dst, rot_coord, TO_UINT(out_val));
			}
			else { write_imagef(dst, rot_coord, out_val); }
		}
	}
}

__kernel void map_rotate(__read_only image2d_t src, __write_only image2d_t dst, sampler_t sampler, 
	int2 out_size, int with_gamma, int2 in_size, int2 src_center, int2 dst_center, double sin_t, double cos_t) {
	int2 coord = (int2)(get_global_id(0), get_global_id(1));
	if (coord.x < out_size.x && coord.y < out_size.y) {
		int2 coord_center = out_size - coord - dst_center - (int2)(1);
		double origin_x = coord_center.x * cos_t + coord_center.y * sin_t;
		double origin_y = coord_center.y * cos_t - coord_center.x * sin_t;
		double2 origin_center = (double2)(src_center.x, src_center.y) - (double2)(origin_x, origin_y);
		if (0.0 <= origin_center.x && origin_center.x < (double)in_size.x &&
			0.0 <= origin_center.y && origin_center.y < (double)in_size.y) {
			float4 out_val = read_imagef(src, sampler, (float2)(origin_x, origin_y));
			if (with_gamma == 1) {
				float4 linear = IS_LINEAR(out_val);
				float4 non_linear = (float4)(1.0f) - linear;
				out_val = non_linear * (1.055f * pow(out_val, POWER) - 0.055f) + linear * 12.92f * out_val;
				out_val *= (float4)(255.0f);
				write_imageui(dst, coord, TO_UINT(out_val));
			}
			else { write_imagef(dst, coord, out_val); }
		}
	}
}