#pragma once

using byte = unsigned char;

struct im_values {

	/* Separate statistics for each channel on index 0..2, the last one for whole image.
	*  Contatains normalised values
	*/
	float mean[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
	float var_sq[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

	bool cached_mean = false;
	bool cached_variance = false;
	bool cached_hist = false;

	/* Separate histograms for each channel on index 0..2, the last one for common historgam.
	*  Four integer arrays of size 256, represents amount of pixels of index value
	*/
	int* histograms[4];

	im_values();

	float* calc_mean(byte* source, size_t pixels);
	float* calc_variance(byte* source, size_t pixels);

	int** calc_historgam(byte* source, size_t pixels);

	void reset();

	~im_values();
};