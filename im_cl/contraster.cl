
#define WRITE_TO_BUF 1
#define EPS 1e-8f


__kernel void manual(__read_only image2d_t src, __global char* buf, __write_only image2d_t dst,
	sampler_t sampler, int2 out_size, int write_mode, float4 factor) {

	int2 out_coord = (int2) (get_global_id(0), get_global_id(1));
	if (out_coord.x < out_size.x && out_coord.y < out_size.y) {
		float4 half_col = (float4)(0.5f);
		float4 in_val = read_imagef(src, sampler, out_coord);
		float4 out_val = factor * (in_val - half_col) + half_col;
		out_val = fmax((float4)0.0f, fmin(1.0f, out_val));

		write_imagef(dst, out_coord, out_val);
		if (write_mode == WRITE_TO_BUF) {
			out_val *= 255.0f;
			int index = 3 * (out_coord.y * out_size.x + out_coord.x);
			buf[index + 0] = (char)rint(out_val.x);
			buf[index + 1] = (char)rint(out_val.y);
			buf[index + 2] = (char)rint(out_val.z);
		}
	}
}


__kernel void exclusive_hist(__read_only image2d_t src, __global char* buf, __write_only image2d_t dst,
	sampler_t sampler, int2 out_size, int write_mode, float4 off, float4 norm) {

	int2 out_coord = (int2) (get_global_id(0), get_global_id(1));
	if (out_coord.x < out_size.x && out_coord.y < out_size.y) {
		float4 in_val = read_imagef(src, sampler, out_coord);
		float4 out_val = (in_val - off) / norm;
		out_val = fmax((float4)0.0f, fmin(1.0f, out_val));

		write_imagef(dst, out_coord, out_val);
		if (write_mode == WRITE_TO_BUF) {
			out_val *= 255.0f;
			int index = 3 * (out_coord.y * out_size.x + out_coord.x);
			buf[index + 0] = (char)rint(out_val.x);
			buf[index + 1] = (char)rint(out_val.y);
			buf[index + 2] = (char)rint(out_val.z);
		}
	}
}

__kernel void adaptive_hist_all(__read_only image2d_t src, __global char* buf,
	__write_only image2d_t dst, sampler_t sampler, int2 out_size, int write_mode, int radius, int exclude) {

	int2 out_coord = (int2) (get_global_id(0), get_global_id(1));
	if (out_coord.x < out_size.x && out_coord.y < out_size.y) {
		short local_hist[256];
		for (int i = 0; i < 256; ++i) { local_hist[i] = 0; }

		for (int y = -radius; y <= radius; ++y) {
			for (int x = -radius; x <= radius; ++x) {
				int2 in_coord = (int2)(out_coord.x + x, out_coord.y + y);
				float4 in_val = rint(read_imagef(src, sampler, in_coord) * 255.0f);
				local_hist[(int)in_val.x]++;
				local_hist[(int)in_val.y]++;
				local_hist[(int)in_val.z]++;
			}
		}

		int min_val = 0, max_val = 255;
		for (int exclude_cnt = exclude; exclude_cnt > 0;
			exclude_cnt -= local_hist[min_val]) { min_val++; }
		for (int exclude_cnt = exclude; exclude_cnt > 0;
			exclude_cnt -= local_hist[max_val]) { max_val--; }

		float4 out_val = read_imagef(src, sampler, out_coord);
		float4 lower = (float4)(min_val / 255.0f);
		float4 upper = (float4)(max_val / 255.0f);
		out_val = (out_val - lower) / (upper - lower + EPS);
		out_val = fmax((float4)0.0f, fmin(1.0f, out_val));

		write_imagef(dst, out_coord, out_val);
		if (write_mode == WRITE_TO_BUF) {
			out_val *= 255.0f;
			int index = 3 * (out_coord.y * out_size.x + out_coord.x);
			buf[index + 0] = (char)rint(out_val.x);
			buf[index + 1] = (char)rint(out_val.y);
			buf[index + 2] = (char)rint(out_val.z);
		}
	}
}


__kernel void adaptive_hist_single(__read_only image2d_t src, __global char* buf,
	__write_only image2d_t dst, sampler_t sampler, int2 out_size, int write_mode, int radius, int exclude) {

	int2 out_coord = (int2) (get_global_id(0), get_global_id(1));
	if (out_coord.x < out_size.x && out_coord.y < out_size.y) {
		short local_hist[256];
		for (int i = 0; i < 256; ++i) { local_hist[i] = 0; }

		for (int y = -radius; y <= radius; ++y) {
			for (int x = -radius; x <= radius; ++x) {
				int2 in_coord = (int2)(out_coord.x + x, out_coord.y + y);
				float4 in_val = rint(read_imagef(src, sampler, in_coord) * 255.0f);
				local_hist[(int)in_val.x]++;
			}
		}

		int min_val = 0, max_val = 255;
		for (int exclude_cnt = exclude; exclude_cnt > 0;
			exclude_cnt -= local_hist[min_val]) {
			min_val++;
		}
		for (int exclude_cnt = exclude; exclude_cnt > 0;
			exclude_cnt -= local_hist[max_val]) {
			max_val--;
		}

		float4 out_val = read_imagef(src, sampler, out_coord);
		float lower = min_val / 255.0f;
		float upper = max_val / 255.0f;
		if (upper - lower > EPS) { out_val.x = (out_val.x - lower) / (upper - lower); }
		out_val = fmax((float4)0.0f, fmin(1.0f, out_val));

		write_imagef(dst, out_coord, out_val);
		if (write_mode == WRITE_TO_BUF) {
			out_val *= 255.0f;
			int index = 3 * (out_coord.y * out_size.x + out_coord.x);
			buf[index + 0] = (char)rint(out_val.x);
			buf[index + 1] = (char)rint(out_val.y);
			buf[index + 2] = (char)rint(out_val.z);
		}
	}
}