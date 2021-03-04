#pragma once

#include"im_object.h"
#include<unordered_map>
#include<iostream>
#include<string>
#include<sstream>

#define POWER_TWO_MAX 20

using command = std::unordered_map<std::string, std::string>;

using functions = std::unordered_map<std::string, cl_kernel>;
using programs = std::unordered_map<std::string, functions>;

using im_ptr = std::shared_ptr<im_object>;


struct util {
	static int power_2_arr[POWER_TWO_MAX];
	static int next_power(int val);

	static void assert_success(cl_int ret_code, const std::string& message);

	static std::string file_ext(std::string filename);

	static command next_action();

	static int euclidean_gcd(size_t a, size_t b);

	static functions map_of(const std::vector<std::string>& func_names);
};