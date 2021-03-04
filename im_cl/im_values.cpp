#include"im_values.h"
#include"im_object.h"

#define GLOBAL 3

im_values::im_values() {
	for (int channel = 0; channel < 4; ++channel) {
		histograms[channel] = new int[256];
		for (int v = 0; v < 256; ++v) { histograms[channel][v] = 0; }
	}
}

im_values::~im_values() {
	for (int channel = 0; channel < 4; ++channel) {
		delete[] histograms[channel];
	}
}

float* im_values::calc_mean(byte* source, size_t pixels) {
	if (cached_mean) { return mean; }
	size_t len = 3 * pixels;
	for (size_t pix = 0; pix < len; pix += 3) {
		for (size_t col = 0; col < 3; ++col) {
			float val = im_object::norm_map[source[pix + col]];
			mean[col] += val; mean[GLOBAL] += val;
		}
	}
	for (size_t col = 0; col < 3; ++col) { mean[col] /= pixels; }
	mean[GLOBAL] /= len; cached_mean = true;
	return mean;
}

float* im_values::calc_variance(byte* source, size_t pixels) {
	if (cached_variance) { return var_sq; }
	calc_mean(source, pixels);
	size_t len = 3 * pixels;
	float diff_sums[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
	for (size_t pix = 0; pix < len; pix += 3) {
		for (size_t col = 0; col < 3; ++col) {
			float val = im_object::norm_map[source[pix + col]];
			float diff_col = val - mean[col], diff_global = val - mean[GLOBAL];
			diff_sums[col] += diff_col * diff_col;
			diff_sums[GLOBAL] += diff_global * diff_global;
		}
	}
	for (size_t col = 0; col < 3; ++col) { var_sq[col] = diff_sums[col] / pixels; }
	var_sq[GLOBAL] = diff_sums[GLOBAL] / len; cached_variance = true;
	return var_sq;
}

int** im_values::calc_historgam(byte* source, size_t pixels) {
	if (cached_hist) { return histograms; }
	size_t len = 3 * pixels;
	for (size_t pix = 0; pix < len; pix += 3) {
		for (size_t col = 0; col < 3; ++col) {
			int val = static_cast<int>(source[pix + col]);
			histograms[col][val]++; histograms[3][val]++;
		}
	}
	return histograms;
}

void im_values::reset() {
	for (size_t channel = 0; channel < 4; ++channel) {
		mean[channel] = 0.0f, var_sq[channel] = 0.0f;
	}
	cached_mean = false, cached_variance = false;
	if (cached_hist) { cached_hist = false;
		for (size_t channel = 0; channel < 4; ++channel) {
			for (int v = 0; v < 256; ++v) { histograms[channel][v] = 0; }
		}
	}
}