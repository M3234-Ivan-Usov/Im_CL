
#define BOARD 0.0031308f
#define POWER (float4)(1.0f / 2.4f)

#define IS_LINEAR(p) (float4)(p.x < BOARD, p.y < BOARD, p.z < BOARD, 0.0f)
#define TO_UINT(p) (uint4)(rint(p.x), rint(p.y), rint(p.z), 0)
#define G(p, n_lin, lin) (n_lin * (1.055f * pow(p, POWER) - 0.055f) + lin * 12.92f * p)

#define WRITE_TO_BUF 1


__kernel void gamma_correction(__read_only image2d_t src, __global char* buf,
	__write_only image2d_t dst, sampler_t sampler, int2 out_size, int write_mode) {

	int2 coord = (int2)(get_global_id(0), get_global_id(1));
	if (coord.x < out_size.x && coord.y < out_size.y) {
		float4 out_val = read_imagef(src, sampler, coord);

		float4 linear = IS_LINEAR(out_val);
		float4 non_linear = (float4)(1.0f) - linear;
		out_val = G(out_val, non_linear, linear) * 255.0f;

		write_imageui(dst, coord, TO_UINT(out_val));
		if (write_mode == WRITE_TO_BUF) {
			int index = 3 * (coord.y * out_size.x + coord.x);
			buf[index + 0] = (char)rint(out_val.x);
			buf[index + 1] = (char)rint(out_val.y);
			buf[index + 2] = (char)rint(out_val.z);
		}
	}
}



/* --- YCbCr conversions ---

   Original YCbCr range:
   16..235 , 16..240, 16..240
*/

__kernel void srgb_to_ycbcr(__read_only image2d_t src, __global char* buf,
	__write_only image2d_t dst, sampler_t sampler, int2 out_size, int write_mode,
	float kr, float kg, float kb) {

	int2 coord = (int2)(get_global_id(0), get_global_id(1));
	if (coord.x < out_size.x && coord.y < out_size.y) {
		float4 in_val = read_imagef(src, sampler, coord);
		float4 out_val = (float4)(0.0f);

		out_val.x = in_val.x * kr + in_val.y * kg + in_val.z * kb;
		out_val.y = (in_val.z - out_val.x) / (2 - 2 * kb);
		out_val.z = (in_val.x - out_val.x) / (2 - 2 * kr);
		out_val.yz += 0.5f;
		out_val = fmax((float4)0.0f, fmin(1.0f, out_val));

		write_imagef(dst, coord, out_val);
		if (write_mode == WRITE_TO_BUF) {
			out_val *= 255.0f;
			int index = 3 * (coord.y * out_size.x + coord.x);
			buf[index + 0] = (char)rint(out_val.x);
			buf[index + 1] = (char)rint(out_val.y);
			buf[index + 2] = (char)rint(out_val.z);
		}
	}
}

__kernel void ycbcr_to_srgb(__read_only image2d_t src, __global char* buf,
	__write_only image2d_t dst, sampler_t sampler, int2 out_size, int write_mode,
	float kr, float kg, float kb) {

	int2 coord = (int2)(get_global_id(0), get_global_id(1));
	if (coord.x < out_size.x && coord.y < out_size.y) {
		float4 in_val = read_imagef(src, sampler, coord);

		in_val.yz -= 0.5f;
		float4 out_val = (float4)(0.0f);
		float kr_mul = 2 - 2 * kr, kb_mul = 2 - 2 * kb;

		out_val.x = in_val.x + kr_mul * in_val.z;
		out_val.y = in_val.x + (kb / kg) * kb_mul * in_val.y + (kr / kg) * kr_mul * in_val.z;
		out_val.z = in_val.x + kb_mul * in_val.y;
		out_val = fmax((float4)0.0f, fmin(1.0f, out_val));

		write_imagef(dst, coord, out_val);
		if (write_mode == WRITE_TO_BUF) {
			out_val *= 255.0f;
			int index = 3 * (coord.y * out_size.x + coord.x);
			buf[index + 0] = (char)rint(out_val.x);
			buf[index + 1] = (char)rint(out_val.y);
			buf[index + 2] = (char)rint(out_val.z);
		}
	}
}



/* --- HSx conversions ---

   Original HSx range:
   0..360, 0..1, 0..1
*/

__kernel void srgb_to_hsv(__read_only image2d_t src, __global char* buf,
	__write_only image2d_t dst, sampler_t sampler, int2 out_size, int write_mode) {

	int2 coord = (int2)(get_global_id(0), get_global_id(1));
	if (coord.x < out_size.x && coord.y < out_size.y) {
		float4 in_val = read_imagef(src, sampler, coord);
		float maximal = fmax(in_val.x, fmax(in_val.y, in_val.z));
		float chroma = maximal - fmin(in_val.x, fmin(in_val.y, in_val.z));
		float4 out_val = (float4)(0.0f);
		if (chroma != 0.0f) {
			if (maximal == in_val.x) { out_val.x = (in_val.y - in_val.z) / chroma; }
			else if (maximal == in_val.y) { out_val.x = 2.0f + (in_val.z - in_val.x) / chroma; }
			else if (maximal == in_val.z) { out_val.x = 4.0f + (in_val.x - in_val.y) / chroma; }
			out_val.x /= 6.0f;
		}
		out_val.y = (maximal == 0.0f) ? 0.0f : chroma / maximal;
		out_val.z = maximal;
		out_val = fmax((float4)0.0f, fmin(1.0f, out_val));

		write_imagef(dst, coord, out_val);
		if (write_mode == WRITE_TO_BUF) {
			out_val *= 255.0f;
			int index = 3 * (coord.y * out_size.x + coord.x);
			buf[index + 0] = (char)rint(out_val.x);
			buf[index + 1] = (char)rint(out_val.y);
			buf[index + 2] = (char)rint(out_val.z);
		}
	}
}

__kernel void hsv_to_srgb(__read_only image2d_t src, __global char* buf,
	__write_only image2d_t dst, sampler_t sampler, int2 out_size, int write_mode) {

	int2 coord = (int2)(get_global_id(0), get_global_id(1));
	if (coord.x < out_size.x && coord.y < out_size.y) {
		float4 in_val = read_imagef(src, sampler, coord);
		in_val.x *= 360.0f;

		float4 col_args = (float4)(5.0f, 3.0f, 1.0f, 0.0f);
		float4 k = fmod(col_args + (float4)(in_val.x / 60.0f), (float4)(6.0f));
		float4 t = fmin(k, fmin((float4)(4.0f) - k, (float4)(1.0f)));
		float4 out_val = (float4)(in_val.z) - in_val.z * in_val.y * fmax((float4)(0.0f), t);
		out_val = fmax((float4)0.0f, fmin(1.0f, out_val));

		write_imagef(dst, coord, out_val);
		if (write_mode == WRITE_TO_BUF) {
			out_val *= 255.0f;
			int index = 3 * (coord.y * out_size.x + coord.x);
			buf[index + 0] = (char)rint(out_val.x);
			buf[index + 1] = (char)rint(out_val.y);
			buf[index + 2] = (char)rint(out_val.z);
		}
	}
}

__kernel void srgb_to_hsl(__read_only image2d_t src, __global char* buf,
	__write_only image2d_t dst, sampler_t sampler, int2 out_size, int write_mode) {

	int2 coord = (int2)(get_global_id(0), get_global_id(1));
	if (coord.x < out_size.x && coord.y < out_size.y) {
		float4 in_val = read_imagef(src, sampler, coord);

		float maximal = fmax(in_val.x, fmax(in_val.y, in_val.z));
		float minimal = fmin(in_val.x, fmin(in_val.y, in_val.z));
		float chroma = maximal - minimal;
		float light = maximal - chroma / 2.0f;
		float4 out_val = (float4)(0.0f);
		if (maximal != minimal) {
			if (maximal == in_val.x) { out_val.x = 0.0f + (in_val.y - in_val.z) / chroma; }
			else if (maximal == in_val.y) { out_val.x = 2.0f + (in_val.z - in_val.x) / chroma; }
			else if (maximal == in_val.z) { out_val.x = 4.0f + (in_val.x - in_val.y) / chroma; }
			out_val.x /= 6.0f;
		}
		if (light != 0.0f && light != 1.0f) { out_val.y = (maximal - light) / fmin(light, 1.0f - light); }
		out_val.z = light;
		out_val = fmax((float4)0.0f, fmin(1.0f, out_val));

		write_imagef(dst, coord, out_val);
		if (write_mode == WRITE_TO_BUF) {
			out_val *= 255.0f;
			int index = 3 * (coord.y * out_size.x + coord.x);
			buf[index + 0] = (char)rint(out_val.x);
			buf[index + 1] = (char)rint(out_val.y);
			buf[index + 2] = (char)rint(out_val.z);
		}
	}
}

__kernel void hsl_to_srgb(__read_only image2d_t src, __global char* buf,
	__write_only image2d_t dst, sampler_t sampler, int2 out_size, int write_mode) {

	int2 coord = (int2)(get_global_id(0), get_global_id(1));
	if (coord.x < out_size.x && coord.y < out_size.y) {
		float4 in_val = read_imagef(src, sampler, coord);
		in_val.x *= 360.0f;

		float4 col_args = (float4)(0.0f, 8.0f, 4.0f, 0.0f);
		float4 k = fmod(col_args + (float4)(in_val.x / 30.0f), (float4)(12.0f));
		float4 a = (float4)(in_val.y) * fmin(in_val.z, 1.0f - in_val.z);
		float4 t = fmin(k - (float4)(3.0f), fmin((float4)(9.0f) - k, (float4)(1.0f)));
		float4 out_val = (float4)(in_val.z) - a * fmax((float4)(-1.0f), t);
		out_val = fmax((float4)0.0f, fmin(1.0f, out_val));

		write_imagef(dst, coord, out_val);
		if (write_mode == WRITE_TO_BUF) {
			out_val *= 255.0f;
			int index = 3 * (coord.y * out_size.x + coord.x);
			buf[index + 0] = (char)rint(out_val.x);
			buf[index + 1] = (char)rint(out_val.y);
			buf[index + 2] = (char)rint(out_val.z);
		}
	}
}

__kernel void hsl_to_hsv(__read_only image2d_t src, __global char* buf,
	__write_only image2d_t dst, sampler_t sampler, int2 out_size, int write_mode) {

	int2 coord = (int2)(get_global_id(0), get_global_id(1));
	if (coord.x < out_size.x && coord.y < out_size.y) {
		float4 in_val = read_imagef(src, sampler, coord);
		float4 out_val = (float4)(0.0f);
		out_val.x = in_val.x;
		out_val.z = in_val.z + in_val.y * fmin(in_val.z, 1.0f - in_val.z);
		if (out_val.z != 0.0f) { out_val.y = 2 * (1.0f - in_val.z / out_val.z); }
		out_val = fmax((float4)0.0f, fmin(1.0f, out_val));

		write_imagef(dst, coord, out_val);
		if (write_mode == WRITE_TO_BUF) {
			out_val *= 255.0f;
			int index = 3 * (coord.y * out_size.x + coord.x);
			buf[index + 0] = (char)rint(out_val.x);
			buf[index + 1] = (char)rint(out_val.y);
			buf[index + 2] = (char)rint(out_val.z);
		}
	}
}

__kernel void hsv_to_hsl(__read_only image2d_t src, __global char* buf,
	__write_only image2d_t dst, sampler_t sampler, int2 out_size, int write_mode) {

	int2 coord = (int2)(get_global_id(0), get_global_id(1));
	if (coord.x < out_size.x && coord.y < out_size.y) {
		float4 in_val = read_imagef(src, sampler, coord);
		float4 out_val = (float4)(0.0f);
		out_val.x = in_val.x;
		out_val.z = in_val.z * (1.0f - in_val.y / 2.0f);
		if (out_val.z != 0.0f && out_val.z != 1.0f) { 
			out_val.y = (in_val.z - out_val.z) / fmin(1.0f - out_val.z, out_val.z);
		}
		out_val = fmax((float4)0.0f, fmin(1.0f, out_val));

		write_imagef(dst, coord, out_val);
		if (write_mode == WRITE_TO_BUF) {
			out_val *= 255.0f;
			int index = 3 * (coord.y * out_size.x + coord.x);
			buf[index + 0] = (char)rint(out_val.x);
			buf[index + 1] = (char)rint(out_val.y);
			buf[index + 2] = (char)rint(out_val.z);
		}
	}
}



/* --- CIE conversions ---

   Original XYZ range:
   0..0.9505, 0..1, 0..1.0890

   Original L*a*b* range:
   0..100, -500..500, -200..200
*/

__kernel void srgb_to_ciexyz(__read_only image2d_t src, __global char* buf,
	__write_only image2d_t dst, sampler_t sampler, int2 out_size, int write_mode) {

	int2 coord = (int2)(get_global_id(0), get_global_id(1));
	if (coord.x < out_size.x && coord.y < out_size.y) {
		float4 in_val = read_imagef(src, sampler, coord);

		float board = 0.04045f;
		float4 non_linear = (float4)(in_val.x > board, in_val.y > board, in_val.z > board, 0.0f);
		float4 linear = (float4)(1.0f) - non_linear, out_val = (float4)(0.0f);
		in_val = non_linear * pow((in_val + 0.055f) / 1.055f, 2.4f) + linear * in_val / 12.92f;
		out_val.x = in_val.x * 0.4124f + in_val.y * 0.3576f + in_val.z * 0.1805f;
		out_val.y = in_val.x * 0.2126f + in_val.y * 0.7152f + in_val.z * 0.0722f;
		out_val.z = in_val.x * 0.0193f + in_val.y * 0.1192f + in_val.z * 0.9505f;
		out_val.x /= 0.9505f; out_val.z /= 1.0890f;
		out_val = fmax((float4)0.0f, fmin(1.0f, out_val));

		write_imagef(dst, coord, out_val);
		if (write_mode == WRITE_TO_BUF) {
			out_val *= 255.0f;
			int index = 3 * (coord.y * out_size.x + coord.x);
			buf[index + 0] = (char)rint(out_val.x);
			buf[index + 1] = (char)rint(out_val.y);
			buf[index + 2] = (char)rint(out_val.z);
		}
	}
}

__kernel void ciexyz_to_srgb(__read_only image2d_t src, __global char* buf,
	__write_only image2d_t dst, sampler_t sampler, int2 out_size, int write_mode) {
	int2 coord = (int2)(get_global_id(0), get_global_id(1));
	if (coord.x < out_size.x && coord.y < out_size.y) {
		float4 in_val = read_imagef(src, sampler, coord);
		float4 out_val = (float4)(0.0f);
		out_val.x = in_val.x * 3.2406f + in_val.y * -1.5372f + in_val.z * -0.4986f;
		out_val.y = in_val.x * -0.9689f + in_val.y * 1.8758f + in_val.z * 0.0415f;
		out_val.z = in_val.x * 0.0557f + in_val.y * -0.2040f + in_val.z * 1.0570f;

		float board = 0.0031308f; float4 gamma_power = (float4)(1.0f / 2.4f);
		float4 non_linear = (float4)(out_val.x > board, out_val.y > board, out_val.z > board, 0.0f);
		float4 linear = (float4)(1.0f) - non_linear;
		out_val = non_linear * (1.055f * pow(out_val, gamma_power) - 0.055f) + linear * 12.92f * out_val;
		out_val = fmax((float4)0.0f, fmin(1.0f, out_val));

		write_imagef(dst, coord, out_val);
		if (write_mode == WRITE_TO_BUF) {
			out_val *= 255.0f;
			int index = 3 * (coord.y * out_size.x + coord.x);
			buf[index + 0] = (char)rint(out_val.x);
			buf[index + 1] = (char)rint(out_val.y);
			buf[index + 2] = (char)rint(out_val.z);
		}
	}
}


#define DELTA 0.20689655172f
#define _3_mul_DELTA_SQR 0.12841854933f
#define DELTA_CUBE 0.00885645167f
#define _4_div_29 0.13793103448f

__kernel void ciexyz_to_cielab(__read_only image2d_t src, __global char* buf,
	__write_only image2d_t dst, sampler_t sampler, int2 out_size, int write_mode) {
	int2 coord = (int2)(get_global_id(0), get_global_id(1));
	if (coord.x < out_size.x && coord.y < out_size.y) {
		float4 in_val = read_imagef(src, sampler, coord);
		float4 linear = (float4)(in_val.x < DELTA_CUBE, in_val.y < DELTA_CUBE, in_val.z < DELTA_CUBE, 0.0f);
		float4 f_val =  (1.0f - linear) * cbrt(in_val) + linear * (in_val / _3_mul_DELTA_SQR + _4_div_29);

		float4 out_val = (float4)(0.0f);
		out_val.x = 1.16f * f_val.y - 0.16f; // origin: 116 * f(y_norm) - 16
		out_val.y = f_val.x - f_val.y; // origin: 500 * f(x_norm - y_norm)
		out_val.z = f_val.y - f_val.z; // origin: 200 * f(y_norm - z_norm)
		out_val.yz = 0.5f * out_val.yz + 0.5f;
		out_val = fmax((float4)0.0f, fmin(1.0f, out_val));

		write_imagef(dst, coord, out_val);
		if (write_mode == WRITE_TO_BUF) {
			out_val *= 255.0f;
			int index = 3 * (coord.y * out_size.x + coord.x);
			buf[index + 0] = (char)rint(out_val.x);
			buf[index + 1] = (char)rint(out_val.y);
			buf[index + 2] = (char)rint(out_val.z);
		}
	}
}

__kernel void cielab_to_ciexyz(__read_only image2d_t src, __global char* buf,
	__write_only image2d_t dst, sampler_t sampler, int2 out_size, int write_mode) {
	int2 coord = (int2)(get_global_id(0), get_global_id(1));
	if (coord.x < out_size.x && coord.y < out_size.y) {
		float4 in_val = read_imagef(src, sampler, coord);
		float4 off_val = (float4)(0.0f);
		off_val.y = (in_val.x + 0.16) / 1.16f; // origin: (L* + 16) / 116
		off_val.x = off_val.y + 2.0f * (in_val.y - 0.5f); // origin : (L* + 16) / 116 + a* / 500
		off_val.z = off_val.y - 2.0f * (in_val.z - 0.5f); // origin :  (L* + 16) / 116 = b* / 200
		float4 linear = (float4)(off_val.x < DELTA_CUBE, off_val.y < DELTA_CUBE, off_val.z < DELTA_CUBE, 0.0f);
		float4 out_val = (1.0f - linear) * off_val * off_val * off_val + linear * (_3_mul_DELTA_SQR * (off_val - _4_div_29));

		write_imagef(dst, coord, out_val);
		if (write_mode == WRITE_TO_BUF) {
			out_val *= 255.0f;
			int index = 3 * (coord.y * out_size.x + coord.x);
			buf[index + 0] = (char)rint(out_val.x);
			buf[index + 1] = (char)rint(out_val.y);
			buf[index + 2] = (char)rint(out_val.z);
		}
	}
}
