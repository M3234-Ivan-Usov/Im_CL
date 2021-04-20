#pragma once
#include<CL/cl.h>
#include<stdexcept>
#include"hardware.h"
#include<array>

struct util;
using histogram = std::array<std::array<int, 256>, 4>;
using channels = std::array<char*, 3>;

#define GAMMA_CORRECTION_ON 1
#define GAMMA_CORRECTION_OFF 0

struct im_object {
	cl_int2 size;
	size_t alloc_size;

	hardware* env;
	cl_mem cl_storage = nullptr;
	char* host_ptr = nullptr;

	
	/* Construct empty image of given size, allocate non-empty buffer if needed */
	im_object(cl_int2 size, hardware* env, cl_mem storage = nullptr);

	/* Construct image with content of 3-channel host_ptr, allocate empty buffer, keeps host_ptr */
	im_object(char* host_ptr, size_t width, size_t height, hardware* env, int direct_gamma);


	im_object(im_object&& other) noexcept;

	/*
	*  Return host_ptr if has one or enqueue buffer read for it.
	*  Return pointer to sequence [ ... [pix.ch0 pix.ch1 pix.ch2] ... ]
	*/
	char* get_host_ptr(int inverse_gamma);
	
	/*
	*  Return array of 3 pointers, each points to sequence [ ... pix.chx pix.chx ... ]
	*/
	channels get_channels(int inverse_gamma);

	histogram calc_histograms(int inverse_gamma);

	/*int** historgrams();
	std::pair<float*, float*> stat();*/

	~im_object();
};
