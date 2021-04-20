
#define INVERSE_BOARD 0.0031308f
#define DIRECT_BOARD 0.04045f

#define LESS(x, y) ((x < y)? 1.0f : 0.0f)


__kernel void denormalise(__read_only image2d_t src, sampler_t sampler,
	__global uchar* dst, int2 size, int gamma_correction) {

	int2 coord = (int2)(get_global_id(0), get_global_id(1));
	float3 in_val = read_imagef(src, sampler, coord).xyz;
	if (gamma_correction == 1) {
		float3 linear = (float3)(LESS(in_val.x, INVERSE_BOARD),
			LESS(in_val.x, INVERSE_BOARD), LESS(in_val.x, INVERSE_BOARD));
		float3 non_linear = (float3)(1.0f) - linear;
		in_val = linear * 12.92f * in_val +
			non_linear * (1.055f * pow(in_val, 1.0f / 2.40f) - 0.055f);
	}
	uchar3 out_val = convert_uchar3(rint(in_val * 255.0f));
	vstore3(out_val, coord.x + size.x * coord.y, dst);
}

__kernel void normalise(__global uchar* src, int2 size,
	__write_only image2d_t dst, int gamma_correction) {

	int2 coord = (int2)(get_global_id(0), get_global_id(1));
	uchar3 byte_val = vload3(coord.x + size.x * coord.y, src);
	float3 out_val = convert_float3(byte_val) / 255.0f;
	if (gamma_correction == 1) {
		float3 linear = (float3)(LESS(out_val.x, DIRECT_BOARD),
			LESS(out_val.y, DIRECT_BOARD), LESS(out_val.z, DIRECT_BOARD));
		float3 non_linear = (float3)(1.0f) - linear;
		out_val = linear * out_val / 12.92f + 
			non_linear * pow((out_val + 0.055f) / 1.055f, 2.4f);
	}
	write_imagef(dst, coord, (float4)(out_val, 0.0f));
}

__kernel void split_channels(__read_only image2d_t src, sampler_t sampler,
	__global uchar* seq_channels, int2 size, int gamma_correction) {

	int2 coord = (int2)(get_global_id(0), get_global_id(1));
	float3 in_val = read_imagef(src, sampler, coord).xyz;
	if (gamma_correction == 1) {
		float3 linear = (float3)(LESS(in_val.x, INVERSE_BOARD),
			LESS(in_val.x, INVERSE_BOARD), LESS(in_val.x, INVERSE_BOARD));
		float3 non_linear = (float3)(1.0f) - linear;
		in_val = linear * 12.92f * in_val +
			non_linear * (1.055f * pow(in_val, 1.0f / 2.40f) - 0.055f);
	}
	uchar3 out_val = convert_uchar3(rint(in_val * 255.0f));
	int linear_coord = coord.x + size.x * coord.y;
	seq_channels[linear_coord] = out_val.x;
	seq_channels[linear_coord + size.x * size.y] = out_val.y;
	seq_channels[linear_coord + 2 * size.x * size.y] = out_val.z;
}
