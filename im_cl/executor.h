#pragma once
#include "util.h"
#include "hardware.h"

struct executor {
	/* Instance of software CL stuff, provided from app */
	hardware* env;

	/* Set of appropriate kernels for current executor */
	functions* kernels;

	executor(hardware* env, functions* kernels);

	cl_int set_common_args(cl_kernel kern, cl_mem src, cl_sampler sampler, cl_mem dst);

	/* Run kernel and immediately call clFinish(). If prev_event != nullptr, waits for its finish before execution */
	virtual void run_blocking(cl_kernel kern, cl_int2 size, cl_event* prev_event = nullptr);

	/* Run kernel and immediately returns with created event. If prev_event != nullptr, waits for its finish before execution */
	virtual cl_event run_with_event(cl_kernel kern, cl_int2 size, cl_event* prev_event = nullptr);

	virtual ~executor() = default;
};