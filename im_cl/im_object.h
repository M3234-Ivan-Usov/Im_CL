#pragma once
#include<CL/cl.h>
#include<stdexcept>
#include"im_values.h"

#define NO_BUFFER_WRITE 0
#define WRITE_TO_BUFFER 1

struct util;

struct im_object {
	/* Gamma correction stuff */
	static float gamma_map[256];
	static float norm_map[256];
	static constexpr float linear_gamma = 0.04045f;
	static constexpr float digital_gamma = 0.0031308f;

	static cl_mem empty_buffer;

	/*  Floating point four channel image on device.
	    Used inside kernel functions.
		Must be linearised except conversions transform 
	*/
	cl_mem cl_im = nullptr;

	/* Byte three channel buffer on device. 
	   Used to read into char* host pointer.
	   Must be converted into non linear space
	*/
	cl_mem cl_storage = empty_buffer;
	bool is_empty_buffer = true;

	char* host_ptr = nullptr;
	cl_command_queue queue;

	cl_sampler sampler;
	cl_image_desc descriptor; 
	cl_image_format format;

	cl_int2 size;

	/* Some image statistics values.
	   Some values are calculated while normalising input image 
	*/
	im_values* values = nullptr;
	
	/* Construct empty image of given size, allocate non-empty buffer if needed */
	im_object(cl_int2 size, cl_context context, 
		cl_command_queue queue, int write_mode, bool def_sampler = true);

	/* Construct image with content of 3-channel host_ptr, allocate empty buffer, keeps host_ptr */
	im_object(char* host_ptr, size_t width, size_t height, cl_context context,
		cl_command_queue queue, bool convert_to_linear, bool def_sampler = true);

	im_object(im_object&& other);

	/* Returns 4-channel ptr with normalised coordinates, gamma corrected if needed */
	float* extend_channels(byte* src, float* extension_map, size_t pixels);

	/* Return host_ptr if has one or queue for one to buffer */
	char* get_host_ptr();

	int** historgrams();
	std::pair<float*, float*> stat();

	~im_object();

private:
	void init_image_stuff();
	void create_mem(float* rgba_ptr, cl_context context, cl_mem_flags flags, bool def_sampler);
};