#pragma once
#include"executor.h"
#include<unordered_map>
#include<set>


/* --- Transforms image colour space ---
*  sRGB, YCbCr601, YCbCr709, HSV, HSL, CIEXYZ
*/
struct converser : public executor { 
	using col_pair = std::pair<std::string, std::string>;

	static std::unordered_map<std::string, std::set<std::string>> conversions;
	static std::unordered_map<std::string, cl_float3> ycc_params;

	converser(hardware* env, functions* conversers);

	/* If src and dst are not a srgb, execute two conversions via srgb.
	Requires image without gamma correction and returns also an image without gamma correction */
	im_ptr run(col_pair colours, im_ptr& src);

private:

	void set_args(cl_kernel kern, im_ptr& src, im_ptr& dst);

	im_ptr run_ycbcr(col_pair colours, im_ptr& src);
};



struct contraster : public executor {
	static constexpr int all_channels = 3;
	static constexpr int single_channel = 0;

	contraster(hardware* env, functions* kernels);

	im_ptr manual(im_ptr& src, float contrast, int channel_mode);
	im_ptr exclusive_hist(im_ptr& src, float exclusive, int channel_mode);
	im_ptr adaptive_hist(im_ptr& src, cl_int2 region, int exclude, int channel_mode);


private:
	void set_args(cl_kernel kern, const im_ptr& src, im_ptr& dst);
};



/* --- Some filters based on convolution ---
*  Gauss blur
*/
struct filter : public executor {
	filter(hardware* env, functions* filters);

	im_ptr gauss(float sigma, int lin_size, im_ptr& src);

private:
	im_ptr convolve(cl_mem conv_kern, im_ptr& src, cl_int radius);
};



/* --- Rotates image on arbitrary angle ---*/
struct rotator : public executor {
	rotator(hardware* env, functions* kernels);

	/* theta -> [-180 .. 180] */
	im_ptr run(const std::string& algo, double theta, const cl_int2& center, im_ptr& src);

	im_ptr simple_angle(const std::string& direction, im_ptr& src);

private:
	cl_int2 rotate_size(const im_ptr& src, double theta);

	std::pair<cl_int2, cl_int2> calc_corners(const cl_int2& rot_size,
		const cl_int2& src_size, const cl_int2& center, double theta);

	cl_int2 rotate_point(const cl_int2& point, const cl_int2& center, double theta);
};



/* --- Denoisoning via Discrete Wavelet Transform --- 
*  Haar basis
*/
struct wavelet : public executor {

	wavelet(hardware* env, functions* wavelets);

	im_ptr run(const std::string& basis, float threshold, const im_ptr& src);

	void set_args(cl_kernel kern, cl_mem src, cl_sampler sampler,
		cl_mem dst, cl_int2 cur_size, int direction);
};



/* --- Upscales and downscales stairs-way ---
*  Bilinear, Lanczos[3-5], Mitchell, Catmull, Adobe, B-Spline, Precise
*/
struct zoomer : public executor {
	zoomer(hardware* env, functions* conv_kernel);

	im_ptr run(const std::string& kernel_type, float factor, im_ptr& src);

	/* Call in case precise output size specified */
	im_ptr precise(im_ptr& src, cl_int2 new_size);

	~zoomer();
private:
	/* Pre-allocated buffers used by BC-Splines*/
	cl_mem polynomials[4][2];

	/* Calculate BC polynomial with respect to given B and C */
	std::pair<cl_mem, cl_mem> calc_spline_polynom(float B, float C);

	/* Set args depending on kernel type */
	void set_args(cl_kernel kern, cl_mem src, cl_mem dst,
		cl_sampler sampler, float factor, int* params);
};