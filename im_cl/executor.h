#pragma once
#include "util.h"

struct executor {
	/* Instance of software CL stuff, provided from app */
	cl_context context;
	cl_command_queue queue;
	functions* kernels;

	executor(cl_context context, cl_command_queue queue, functions* kernels);

	/* Many kernels requires at least 6 params: src image, dst buf, dst image, src sampler, out_size, write_mode */
	virtual void set_common_args(cl_kernel kern, const im_object& src, const im_object& dst, int write_mode);

	/* Run kernel and immediately call clFinish(). If prev_event != nullptr, waits for its finish before execution */
	virtual void run_blocking(cl_kernel kern, cl_int2 size, cl_event* prev_event = nullptr);

	/* Run kernel and immediately returns with created event. If prev_event != nullptr, waits for its finish before execution */
	virtual cl_event run_with_event(cl_kernel kern, cl_int2 size, cl_event* prev_event = nullptr);

	virtual ~executor() = default;
};