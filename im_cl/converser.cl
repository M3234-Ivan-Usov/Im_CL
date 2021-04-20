
/* --- YCbCr conversions ---

   Original YCbCr range:
   16..235 , 16..240, 16..240
*/

__kernel void srgb_to_ycbcr(__read_only image2d_t src, sampler_t sampler,
	__write_only image2d_t dst, float3 params) {
	int2 coord = (int2)(get_global_id(0), get_global_id(1));
	float4 in_val = read_imagef(src, sampler, coord);
	float4 out_val = 0.0f;
	out_val.x = in_val.x * params.x + in_val.y * params.y + in_val.z * params.z;
	out_val.y = (in_val.z - out_val.x) / (2 - 2 * params.z) + 0.5f;
	out_val.z = (in_val.x - out_val.x) / (2 - 2 * params.x) + 0.5f;
	write_imagef(dst, coord, fmax((float4)0.0f, fmin(1.0f, out_val)));
}

__kernel void ycbcr_to_srgb(__read_only image2d_t src, sampler_t sampler,
	__write_only image2d_t dst, float3 params) {
	int2 coord = (int2)(get_global_id(0), get_global_id(1));
	float4 in_val = read_imagef(src, sampler, coord);
	in_val.yz -= 0.5f;
	float3 mul = 2.0f - 2.0f * params;
	float4 out_val = (float4)(in_val.x);
	out_val.x += mul.x * in_val.z;
	out_val.y += (params.z / params.y) * mul.z * in_val.y + (params.x / params.y) * mul.x * in_val.z;
	out_val.z += mul.z * in_val.y;
	write_imagef(dst, coord, fmax((float4)0.0f, fmin(1.0f, out_val)));
}



/* --- HSx conversions ---

   Original HSx range:
   0..360, 0..1, 0..1
*/

__kernel void srgb_to_hsv(__read_only image2d_t src, sampler_t sampler, __write_only image2d_t dst) {
	int2 coord = (int2)(get_global_id(0), get_global_id(1));
	float4 in_val = read_imagef(src, sampler, coord);
	float4 out_val = 0.0f;
	float maximal = fmax(in_val.x, fmax(in_val.y, in_val.z));
	float chroma = maximal - fmin(in_val.x, fmin(in_val.y, in_val.z));
	if (chroma != 0.0f) {
		if (maximal == in_val.x) { out_val.x = (in_val.y - in_val.z) / chroma; }
		else if (maximal == in_val.y) { out_val.x = 2.0f + (in_val.z - in_val.x) / chroma; }
		else if (maximal == in_val.z) { out_val.x = 4.0f + (in_val.x - in_val.y) / chroma; }
		out_val.x /= 6.0f;
	}
	out_val.y = (maximal == 0.0f) ? 0.0f : chroma / maximal;
	out_val.z = maximal;
	write_imagef(dst, coord, fmax((float4)0.0f, fmin(1.0f, out_val)));
}

__kernel void hsv_to_srgb(__read_only image2d_t src, sampler_t sampler, __write_only image2d_t dst) {
	int2 coord = (int2)(get_global_id(0), get_global_id(1));
	float4 in_val = read_imagef(src, sampler, coord);
	in_val.x *= 360.0f;
	float4 col_args = (float4)(5.0f, 3.0f, 1.0f, 0.0f);
	float4 k = fmod(col_args + (float4)(in_val.x / 60.0f), (float4)(6.0f));
	float4 t = fmin(k, fmin((float4)(4.0f) - k, (float4)(1.0f)));
	float4 out_val = (float4)(in_val.z) - in_val.z * in_val.y * fmax((float4)(0.0f), t);
	write_imagef(dst, coord, fmax((float4)0.0f, fmin(1.0f, out_val)));
}

__kernel void srgb_to_hsl(__read_only image2d_t src, sampler_t sampler, __write_only image2d_t dst) {
	int2 coord = (int2)(get_global_id(0), get_global_id(1));
	float4 in_val = read_imagef(src, sampler, coord);
	float4 out_val = 0.0f;
	float maximal = fmax(in_val.x, fmax(in_val.y, in_val.z));
	float minimal = fmin(in_val.x, fmin(in_val.y, in_val.z));
	float chroma = maximal - minimal;
	float light = maximal - chroma / 2.0f;
	if (maximal != minimal) {
		if (maximal == in_val.x) { out_val.x = 0.0f + (in_val.y - in_val.z) / chroma; }
		else if (maximal == in_val.y) { out_val.x = 2.0f + (in_val.z - in_val.x) / chroma; }
		else if (maximal == in_val.z) { out_val.x = 4.0f + (in_val.x - in_val.y) / chroma; }
		out_val.x /= 6.0f;
	}
	if (light != 0.0f && light != 1.0f) { out_val.y = (maximal - light) / fmin(light, 1.0f - light); }
	out_val.z = light;
	write_imagef(dst, coord, fmax((float4)0.0f, fmin(1.0f, out_val)));
}

__kernel void hsl_to_srgb(__read_only image2d_t src, sampler_t sampler, __write_only image2d_t dst) {
	int2 coord = (int2)(get_global_id(0), get_global_id(1));
	float4 in_val = read_imagef(src, sampler, coord);
	in_val.x *= 360.0f;
	float4 col_args = (float4)(0.0f, 8.0f, 4.0f, 4.0f);
	float4 k = fmod(col_args + (float4)(in_val.x / 30.0f), (float4)(12.0f));
	float4 a = (float4)(in_val.y) * fmin(in_val.z, 1.0f - in_val.z);
	float4 t = fmin(k - (float4)(3.0f), fmin((float4)(9.0f) - k, (float4)(1.0f)));
	float4 out_val = (float4)(in_val.z) - a * fmax((float4)(-1.0f), t);
	write_imagef(dst, coord, fmax((float4)0.0f, fmin(1.0f, out_val)));
}

__kernel void hsl_to_hsv(__read_only image2d_t src, sampler_t sampler, __write_only image2d_t dst) {
	int2 coord = (int2)(get_global_id(0), get_global_id(1));
	float4 in_val = read_imagef(src, sampler, coord);
	float4 out_val = 0.0f;
	out_val.x = in_val.x;
	out_val.z = in_val.z + in_val.y * fmin(in_val.z, 1.0f - in_val.z);
	if (out_val.z != 0.0f) { out_val.y = 2 * (1.0f - in_val.z / out_val.z); }
	write_imagef(dst, coord, fmax((float4)0.0f, fmin(1.0f, out_val)));
}

__kernel void hsv_to_hsl(__read_only image2d_t src, sampler_t sampler, __write_only image2d_t dst) {
	int2 coord = (int2)(get_global_id(0), get_global_id(1));
	float4 in_val = read_imagef(src, sampler, coord);
	float4 out_val = 0.0f;
	out_val.x = in_val.x;
	out_val.z = in_val.z * (1.0f - in_val.y / 2.0f);
	if (out_val.z != 0.0f && out_val.z != 1.0f) { 
		out_val.y = (in_val.z - out_val.z) / fmin(1.0f - out_val.z, out_val.z);
	}
	write_imagef(dst, coord, fmax((float4)0.0f, fmin(1.0f, out_val)));
}



/* --- CIE conversions ---

   Original XYZ range:
   0..0.9505, 0..1, 0..1.0890

   Original L*a*b* range:
   0..100, -500..500, -200..200
*/

#define LESS(x, y) ((x < y)? 1.0f : 0.0f)
#define XYZ_NORM (float2)(0.9505f, 1.0890f)

#define srgb_x (float4)(0.4124f, 0.2126f, 0.0193f, 0.0f)
#define srgb_y (float4)(0.3576f, 0.7152f, 0.1192f, 0.0f)
#define srgb_z (float4)(0.1805f, 0.0722f, 0.9505f, 0.0f)
#define XYZ_BOARD 0.04045f

__kernel void srgb_to_ciexyz(__read_only image2d_t src,
	sampler_t sampler, __write_only image2d_t dst) {
	int2 coord = (int2)(get_global_id(0), get_global_id(1));
	float4 in_val = read_imagef(src, sampler, coord);
	float4 linear = (float4)(LESS(in_val.x, XYZ_BOARD),
		LESS(in_val.y, XYZ_BOARD), LESS(in_val.z, XYZ_BOARD), 0.0f);
	float4 non_linear = (float4)1.0f - linear;
	float4 lin_val = non_linear * pow((in_val + 0.055f) / 1.055f, 2.4f) + linear * in_val / 12.92f;
	float4 out_val = lin_val.x * srgb_x + lin_val.y * srgb_y + lin_val.z * srgb_z;
	out_val.xz /= XYZ_NORM;
	write_imagef(dst, coord, fmax((float4)0.0f, fmin(1.0f, out_val)));
}

#define xyz_r (float4)(3.2406f, -0.9689f, 0.0557f, 0.0f)
#define xyz_g (float4)(-1.5372f, 1.8758f, -0.2040f, 0.0f)
#define xyz_b (float4)(-0.4986f, 0.0415f, 1.0570f, 0.0f)
#define SRGB_BOARD 0.0031308f

__kernel void ciexyz_to_srgb(__read_only image2d_t src, sampler_t sampler, __write_only image2d_t dst) {
	int2 coord = (int2)(get_global_id(0), get_global_id(1));
	float4 in_val = read_imagef(src, sampler, coord);
	in_val.xz *= XYZ_NORM;
	float4 out_val = in_val.x * xyz_r + in_val.y * xyz_g + in_val.z * xyz_b;
	float4 linear = (float4)(LESS(out_val.x, SRGB_BOARD),
		LESS(out_val.y, SRGB_BOARD), LESS(out_val.z, SRGB_BOARD), 0.0f);
	float4 non_linear = (float4)1.0f - linear;
	out_val = non_linear * (1.055f * pow(out_val, 1.0f / 2.4f) - 0.055f) + linear * 12.92f * out_val;
	write_imagef(dst, coord, fmax((float4)0.0f, fmin(1.0f, out_val)));
}


#define DELTA 0.20689655172f
#define _3_mul_DELTA_SQR 0.12841854933f
#define DELTA_CUBE 0.00885645167f
#define _4_div_29 0.13793103448f

__kernel void ciexyz_to_cielab(__read_only image2d_t src, sampler_t sampler, __write_only image2d_t dst) {
	int2 coord = (int2)(get_global_id(0), get_global_id(1));
	float4 in_val = read_imagef(src, sampler, coord);
	float4 out_val = 0.0f;
	float4 linear = (float4)(in_val.x < DELTA_CUBE, in_val.y < DELTA_CUBE, in_val.z < DELTA_CUBE, 0.0f);
	float4 f_val =  (1.0f - linear) * cbrt(in_val) + linear * (in_val / _3_mul_DELTA_SQR + _4_div_29);
	out_val.x = 1.16f * f_val.y - 0.16f; // origin: 116 * f(y_norm) - 16
	out_val.y = f_val.x - f_val.y; // origin: 500 * f(x_norm - y_norm)
	out_val.z = f_val.y - f_val.z; // origin: 200 * f(y_norm - z_norm)
	out_val.yz = 0.5f * out_val.yz + 0.5f;
	write_imagef(dst, coord, fmax((float4)0.0f, fmin(1.0f, out_val)));
}

__kernel void cielab_to_ciexyz(__read_only image2d_t src, sampler_t sampler, __write_only image2d_t dst) {
	int2 coord = (int2)(get_global_id(0), get_global_id(1));
	float4 in_val = read_imagef(src, sampler, coord);
	float4 off_val = (float4)(0.0f, (in_val.x + 0.16) / 1.16f, 0.0f, 0.0f); // origin: (L* + 16) / 116
	off_val.x = off_val.y + 2.0f * (in_val.y - 0.5f); // origin : (L* + 16) / 116 + a* / 500
	off_val.z = off_val.y - 2.0f * (in_val.z - 0.5f); // origin :  (L* + 16) / 116 = b* / 200
	float4 linear = (float4)(off_val.x < DELTA_CUBE, off_val.y < DELTA_CUBE, off_val.z < DELTA_CUBE, 0.0f);
	float4 out_val = (1.0f - linear) * off_val * off_val * off_val + linear * (_3_mul_DELTA_SQR * (off_val - _4_div_29));
	write_imagef(dst, coord, fmax((float4)0.0f, fmin(1.0f, out_val)));
}
