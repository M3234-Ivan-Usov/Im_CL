#pragma once
#include"im_object.h"
#include"util.h"
#include<string>
#include<iostream>
#include<fstream>

struct io_manager {
	static im_ptr load_pnm(hardware* env, const std::string& filename, int gamma);

	static void write_pnm(im_ptr& storage, const std::string& filename, int inverse_gamma);
};