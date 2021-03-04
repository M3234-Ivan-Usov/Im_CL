#include"util.h"

int util::power_2_arr[];

int util::next_power(int val) {
	for (int index = 0; index < POWER_TWO_MAX; ++index) {
		if (power_2_arr[index] >= val) { return power_2_arr[index]; }
	}
	throw std::runtime_error("Too large number");
}

void util::assert_success(cl_int ret_code, const std::string& message) {
	if (ret_code == CL_SUCCESS) { return; }
	std::string code;
	switch (ret_code) {
	case CL_INVALID_ARG_SIZE: code = "invalid arg size"; break;
	case CL_INVALID_ARG_VALUE: code = "invalid arg value"; break;
	case CL_INVALID_COMMAND_QUEUE: code = "invalid queue"; break;
	case CL_INVALID_CONTEXT: code = "invalid context"; break;
	case CL_INVALID_DEVICE: code = "invalid device"; break;
	case CL_INVALID_HOST_PTR: code = "invalid host ptr"; break;
	case CL_INVALID_IMAGE_DESCRIPTOR: code = "invalid image descriptor"; break;
	case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: code = "invalid image format descriptor"; break;
	case CL_INVALID_IMAGE_SIZE: code = "invalid image size"; break;
	case CL_INVALID_KERNEL: code = "invalid kernel"; break;
	case CL_INVALID_KERNEL_ARGS: code = "invalid kernel args"; break;
	case CL_INVALID_MEM_OBJECT: code = "invalid mem object"; break;
	case CL_INVALID_OPERATION: code = "invalid operation"; break;
	case CL_INVALID_PLATFORM: code = "invalid platform"; break;
	case CL_INVALID_PROGRAM: code = "invalid program"; break;
	case CL_INVALID_SAMPLER: code = "invalid sampler"; break;
	case CL_INVALID_VALUE: code = "invalid value"; break;
	case CL_MEM_OBJECT_ALLOCATION_FAILURE: code = "mem object allocation failure"; break;
	case CL_OUT_OF_HOST_MEMORY: code = "out of host memory"; break;
	case CL_OUT_OF_RESOURCES: code = "out of resources"; break;
	}
	throw std::runtime_error(message + " (" + code + ")");
}

std::string util::file_ext(std::string filename) {
	size_t point = filename.find_last_of('.');
	return filename.substr(point, filename.length() - point);
}

command util::next_action() {
	using is_it = std::istream_iterator<std::string>;
	std::string cmd, arg("arg"); char free_arg = '0';
	while (true) {
		std::getline(std::cin, cmd);
		std::istringstream iss(cmd);
		std::vector<std::string> seq((is_it(iss)), is_it());
		if (seq.empty()) { std::cout << "> "; continue; }
		command spl_arg = { { "exe", seq[0] } };
		for (size_t p = 1; p < seq.size(); ++p) {
			if (seq[p][0] == '-') { spl_arg.emplace(seq[p], seq[p + 1]); ++p; }
			else { spl_arg.emplace(arg + free_arg++, seq[p]); }
		}
		return spl_arg;
	}
}

functions util::map_of(const std::vector<std::string>& funcs) {
	functions kern_map;
	for (auto name = funcs.begin(); name != funcs.end(); ++name) {
		kern_map[*name] = NULL;
	}
	return kern_map;
}

int util::euclidean_gcd(size_t a, size_t b) {
	while (a != 0 && b != 0) {
		if (a > b) { a %= b; }
		else { b %= b; }
	}
	return static_cast<int>(a + b);
}