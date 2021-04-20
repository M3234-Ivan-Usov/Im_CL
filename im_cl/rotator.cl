
__kernel void counter_clockwise(__read_only image2d_t src, 
	sampler_t sampler, __write_only image2d_t dst, int2 sz) {
	int2 out_cd = (int2)(get_global_id(0), get_global_id(1));
	int2 in_cd = (int2)(out_cd.y, sz.x - out_cd.x - 1);
	write_imagef(dst, out_cd, read_imagef(src, sampler, in_cd));
}

__kernel void clockwise(__read_only image2d_t src,
	sampler_t sampler, __write_only image2d_t dst, int2 sz) {
	int2 out_cd = (int2)(get_global_id(0), get_global_id(1));
	int2 in_cd = (int2)(sz.y - out_cd.y - 1, out_cd.x);
	write_imagef(dst, out_cd, read_imagef(src, sampler, in_cd));
}

/* Nullable nearest sampler */
__kernel void shear(__read_only image2d_t src, sampler_t sampler, __write_only image2d_t dst,
	int2 out_sz, float2 src_center, int2 dst_center, float2 angles) {

	int2 cd = (int2)(get_global_id(0), get_global_id(1));
	int2 cd_center = out_sz - cd - dst_center - (int2)(1);
	float rot_x = cd_center.x + angles.x * cd_center.y;
	float rot_y = cd_center.y + angles.y * rot_x;
	rot_x += angles.x * rot_y;
	float2 rot_cd = src_center - (float2)(rot_x, rot_y);
	write_imagef(dst, cd, read_imagef(src, sampler, rot_cd));
}

/* Nullable bilinear sampler */
__kernel void map(__read_only image2d_t src, sampler_t sampler, __write_only image2d_t dst,
	int2 out_sz, float2 src_center, int2 dst_center, float2 angles) {
	
	int2 cd = (int2)(get_global_id(0), get_global_id(1));
	int2 cd_center = out_sz - cd - dst_center - (int2)(1);
	float2 origin = (float2)(
		cd_center.x * angles.y - cd_center.y * angles.x,
		cd_center.y * angles.y + cd_center.x * angles.x
		);
	float2 rot_cd = src_center - rint(origin);
	write_imagef(dst, cd, read_imagef(src, sampler, rot_cd));
}