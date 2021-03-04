#define _CRT_SECURE_NO_WARNINGS
#include"io_manager.h"

im_ptr io_manager::load_pnm(cl_context context, cl_command_queue queue,
	std::string filename, bool convert_to_linear) {
	FILE* in_image = fopen(filename.c_str(), "rb");
	if (in_image == nullptr) { throw std::runtime_error("Failed to open " + filename); }
	char f[4]; size_t w, h, m;
	int head = fscanf(in_image, "%s\n%u %u\n%u\n", f, &w, &h, &m);
	char* source = new char[3 * w * h];
	fread(source, 1, 3ull * w * h, in_image);
	im_ptr im = std::make_shared<im_object>(source, w, h, context, queue, convert_to_linear);
	fclose(in_image);
	return std::move(im);
}

void io_manager::write_pnm(im_object& im, std::string filename) {
	FILE* out_image = fopen(filename.c_str(), "wb");
	if (out_image == nullptr) { throw std::runtime_error("Failed to open " + filename); }
	fprintf(out_image, "P6\n%d %d\n%d\n", im.size.x, im.size.y, 255);
	char* src = im.get_host_ptr();
	fwrite(src, 1, 3ull * im.size.x * im.size.y, out_image);
	fclose(out_image);
}

