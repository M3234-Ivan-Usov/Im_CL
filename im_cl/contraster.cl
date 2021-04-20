
__kernel void manual(__read_only image2d_t src, sampler_t sampler,
	__write_only image2d_t dst, float4 factor) {
	int2 cd = (int2)(get_global_id(0), get_global_id(1));
	float4 out_val = factor * (read_imagef(src, sampler, cd) - 0.5f) + 0.5f;
	write_imagef(dst, cd, fmax((float4)0.0f, fmin(1.0f, out_val)));
}

__kernel void exclusive_hist(__read_only image2d_t src, sampler_t sampler,
	__write_only image2d_t dst, float4 off, float4 norm) {
	int2 cd = (int2)(get_global_id(0), get_global_id(1));
	float4 out_val = (read_imagef(src, sampler, cd) - off) / norm;
	write_imagef(dst, cd, fmax((float4)0.0f, fmin(1.0f, out_val)));
}


#define ADAPTIVE_EPS 1e-5f

__kernel void adaptive_hist(__read_only image2d_t src, sampler_t sampler,
	__write_only image2d_t dst, int2 sz, int2 radius, int exclude) {
	int2 offset = (int2) (get_global_id(0), get_global_id(1)) * radius;
	short local_hist[256];
	for (int i = 0; i < 256; ++i) { local_hist[i] = 0; }
	for (int y = 0; y < radius.y; ++y) {
		for (int x = 0; x < radius.x; ++x) {
			int2 cur_cd = offset + (int2)(x, y);
			float4 cur_val = read_imagef(src, sampler, cur_cd);
			int4 int_val = convert_int4(cur_val * 255.0f);
			local_hist[int_val.x]++;
			local_hist[int_val.y]++;
			local_hist[int_val.z]++;
		}
	}
	int min_val = 0, max_val = 255;
	for (int exclude_cnt = exclude; exclude_cnt > 0;
		exclude_cnt -= local_hist[min_val]) { min_val++; }
	for (int exclude_cnt = exclude; exclude_cnt > 0; 
		exclude_cnt -= local_hist[max_val]) { max_val--; }

	float min_f = min_val / 255.0f, max_f = max_val / 255.0f;
	float norm = max_f - min_f;
	if (norm < ADAPTIVE_EPS) { min_f = 0.0f, norm = 1.0f; }
	for (int y = 0; y < radius.y; ++y) {
		for (int x = 0; x < radius.x; ++x) {
			int2 cur_cd = offset + (int2)(x, y);
			if (cur_cd.x < sz.x && cur_cd.y < sz.y) {
				float4 out_val = (read_imagef(src, sampler, cur_cd) - min_f) / norm;
				write_imagef(dst, cur_cd, fmax((float4)0.0f, fmin(1.0f, out_val)));
			}
		}
	}
}
