
__kernel void conv_2D(__read_only image2d_t src, sampler_t sampler,
	__write_only image2d_t dst, int radius, __read_only image2d_t kern) {
	int2 cd = (int2) (get_global_id(0), get_global_id(1));
	float4 out_val = (float4)(0.0f);
	for (int y = -radius; y <= radius; ++y) {
		for (int x = -radius; x <= radius; ++x) {
			float4 pix = read_imagef(src, sampler, (int2)(cd.x + x, cd.y + y));
			out_val += pix * read_imagef(kern, sampler, (int2)(x + radius, y + radius)).w;
		}
	}
	write_imagef(dst, cd, fmax((float4)0.0f, fmin(1.0f, out_val)));
}

__kernel void horizontal_conv(__read_only image2d_t src, sampler_t sampler,
	__write_only image2d_t dst, int radius, __constant float* kern) {
	int2 cd = (int2)(get_global_id(0), get_global_id(1));
	float4 out_val = 0.0f;
	for (int x = -radius; x <= radius; ++x) {
		float4 pix = read_imagef(src, sampler, (int2)(cd.x + x, cd.y));
		out_val += pix * kern[x + radius];
	}
	write_imagef(dst, cd, fmax((float4)0.0f, fmin(1.0f, out_val)));
}

__kernel void vertical_conv(__read_only image2d_t src, sampler_t sampler,
	__write_only image2d_t dst, int radius, __constant float* kern) {
	int2 cd = (int2)(get_global_id(0), get_global_id(1));
	float4 out_val = 0.0f;
	for (int y = -radius; y <= radius; ++y) {
		float4 pix = read_imagef(src, sampler, (int2)(cd.x, cd.y + y));
		out_val += pix * kern[y + radius];
	}
	write_imagef(dst, cd, fmax((float4)0.0f, fmin(1.0f, out_val)));
}