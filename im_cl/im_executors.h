#pragma once
#include"executor.h"
#include<set>


/* --- Transforms image colour space ---
*  sRGB, YCbCr601, YCbCr709, HSV, HSL, CIEXYZ
*/
struct converser : public executor {
	converser(cl_context context, cl_command_queue queue, functions* conversers);

	/* If src and dst are not a srgb, execute two conversions via srgb.
	Requires image without gamma correction and returns also an image without gamma correction */
	im_ptr run(const std::string& from_cs, const std::string& to_cs, const im_object& src, int write_mode = WRITE_TO_BUFFER);

private:
	im_ptr from_srgb(const std::string& colour_space, const im_object& src, int write_mode = WRITE_TO_BUFFER);
	im_ptr to_srgb(const std::string& colour_space, const im_object& src, int write_mode = WRITE_TO_BUFFER);

	void set_ycc_args(cl_kernel kern, float* kr, float* kg, float* kb);
};



struct contraster : public executor {
	struct args {
		float contrast_level = 0.0f;
		float exclude = 0.0039f;
		int apt_exclude = 5;
		int apt_radius = 4;
		std::string via_space;
		std::string type;

		args(const std::string& t, const std::string& v,
			const std::string& c, const std::string& e, const std::string& r);
	};

	converser* converser_ptr;
	std::set<std::string> via_spaces = {"hsv", "hsl", "ycc601", "ycc709"};

	contraster(cl_context context, cl_command_queue queue, functions* kernels, converser* converser_ptr);

	im_ptr run(im_object& src, const args& params);

private:
	im_ptr exclusive_hist(im_object& src, float exclusive, int channels);
	im_ptr adaptive_hist(im_object& src, int exclude, int radius, int channels);
	im_ptr manual(im_object& src, float contrast, int channels);
};



/* --- Some filters based on convolution ---
*  Gauss blur
*/
struct filter : public executor {
	filter(cl_context context, cl_command_queue queue, functions* filters);

	/* lin_size - kernel length in one dimension */
	im_ptr gauss(float sigma, int lin_size, im_object& src);
};



/* --- Rotates image on arbitrary angle ---*/
struct rotator : public executor {
	rotator(cl_context context, cl_command_queue queue, functions* kernels);

	/* theta -> [-180 .. 180] */
	im_ptr run(const std::string& algo, double theta, im_object& src);

	~rotator() = default;
private:

	im_ptr simple_angle(int theta, im_object& src);

	cl_int2 rotate_size(const im_object& src, double theta);

	cl_int2 im_center(const im_object& src);

	void set_coord_args(cl_kernel kern, const cl_int2& in_size,
		const cl_int2& src_center, const cl_int2& dst_center);
};



/* --- Denoisoning via Discrete Wavelet Transform --- 
*  Haar basis
*/
struct wavelet : public executor {
	cl_sampler nullable_sampler;
	cl_kernel gamma_corrector;

	wavelet(cl_context context, cl_command_queue queue, functions* wavelets, cl_kernel gamma_corrector);

	im_ptr run(const std::string& basis, float threshold, const im_object& src);

	void set_haar_args(cl_kernel kern, im_object* src, im_object* dst, cl_int2 cur_size, int dim);

	~wavelet();
};



/* --- Upscales and downscales stairs-way ---
*  Bilinear, Lanczos[3-5], Mitchell, Catmull, Adobe, B-Spline, Precise
*/
struct zoomer : public executor {
	zoomer(cl_context context, cl_command_queue queue, functions* conv_kernel);

	im_ptr run(const std::string& kernel_type, float factor, im_object& src);

	~zoomer();
private:
	cl_mem polynomials[4];

	cl_mem calc_spline_polynom(float B, float C);

	void set_extra_args(cl_kernel kern, im_object& src, im_object& dst,
		int func, float factor, int order, cl_mem polynom);
};