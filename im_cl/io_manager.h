#pragma once
#include"im_object.h"
#include"util.h"
#include<string>
#include<iostream>
#include<fstream>

struct io_manager {
	static im_ptr load_pnm(cl_context context, cl_command_queue queue,
		std::string filename, bool convert_to_linear = true);

	static void write_pnm(im_object& storage, std::string filename);
};