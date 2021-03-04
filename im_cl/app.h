#pragma once

#include"hardware.h"
#include"im_executors.h"
#include"io_manager.h"

#include<string>
#include<type_traits>
#include<unordered_map>

using load_fun = im_ptr(*) (cl_context, cl_command_queue, std::string, bool);
using write_fun = void (*) (im_object&, std::string);

struct app {

	/* OpenCL environment */
	hardware env;
	cl_context context;
	cl_command_queue queue;

	/* Gather all program objects into one vector */
	std::vector<cl_program> prog_objects;

	/* Map kernel name to kernel objects */
	programs prog_tree;

	/* Executors themselves */
	zoomer* zoomer_ptr;
	converser* converser_ptr;
	rotator* rotator_ptr;
	filter* filter_ptr;
	wavelet* wavelet_ptr;
	contraster* contraster_ptr;
	

	app(size_t plat_id, size_t dev_id);
	void env_info();

	/* Given filename, creates ready for use read-only im_object */
	im_ptr get_im(std::string filename, bool convert_to_linear = true);

	/* Puts im_object.host_ptr into file */
	void put_im(std::string filename, im_object& im);

	~app();

private:
	void match_extensions();
	void match_kernels();
	void compile_kernels();
	void init_executors();

	std::unordered_map<std::string, load_fun> loader;
	std::unordered_map<std::string, write_fun> writer;
};