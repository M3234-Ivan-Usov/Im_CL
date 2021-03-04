#include"app.h"

app* app_ptr = nullptr;

std::unordered_map<std::string, std::string> cmd_syntax = {
	{"zoom", "zoom <input> -o <output> -f <factor> [-t <type>]"},
	{"init", "init [-p] <platform_id> [-d] <device_id>"},
	{"converse", "converse <input> -o <output> [-t <to_CS>] [-f <from_CS>]"},
	{"rotate", "rotate <input> -o <output> -a <angle> [-t <type>]"},
	{"gauss", "gauss <input> -o <output> [-s <sigma>] [-w <window_size>]"},
	{"contrast", "contrast <input> -o <output> [-t <type>] [-v <via_space>] [-c <contrast>] [-r <radius>] [-e <exclude>"},
	{"wavelet", "wavelet <input> -o <output> [-b <basis>] [-t <threshold>]"}
};

void assert_init() {
	if (app_ptr != nullptr) { return; }
	throw std::runtime_error("Not initialised");
}

void report_cmd(std::string cmd) { std::cerr << "Expected: " << cmd_syntax[cmd] << std::endl; }

void pre_init_static() {
	for (int col = 0; col < 256; ++col) {
		float val = col / 255.0f;
		im_object::gamma_map[col] = (val <= im_object::linear_gamma) ?
			val / 12.92f : powf((val + 0.055f) / 1.055f, 2.4f);
		im_object::norm_map[col] = val;
	}
	for (int index = 0, power_2 = 1; index < POWER_TWO_MAX; ++index, power_2 <<= 1) {
		util::power_2_arr[index] = power_2;
	}
}

int main(int argc, char** argv) {
	pre_init_static();
	try { app_ptr = new app(0, 0); }
	catch (std::runtime_error e) { std::cerr << " Default init failed: " << std::endl << e.what() << std::endl; }
	while (true) {
		std::cout << "> ";
		command cmd = util::next_action();
		try {
			if (cmd["exe"] == "zoom") { assert_init();
				std::string input = cmd["arg0"], output = cmd["-o"];
				std::string scale = cmd["-f"], kern_type = cmd["-t"];
				if (scale.empty() || input.empty() || output.empty()) { 
					report_cmd("zoom"); continue;
				}
				float factor = atoi(scale.c_str()) / 100.0f;
				if (kern_type.empty()) { kern_type = "lan3"; }
				im_ptr src = app_ptr->get_im(input);
				im_ptr scaled = app_ptr->zoomer_ptr->run(kern_type, factor, *src);
				app_ptr->put_im(output, *scaled);
			}
			else if (cmd["exe"] == "converse") { assert_init();
				std::string input = cmd["arg0"], output = cmd["-o"];
				std::string to_cs = cmd["-t"], from_cs = cmd["-f"];
				if ((to_cs.empty() && from_cs.empty()) || input.empty() || output.empty()) {
					report_cmd("converse"); continue;
				}
				if (from_cs.empty()) { from_cs = "srgb"; }
				else if (to_cs.empty()) { to_cs = "srgb"; }
				im_ptr src = app_ptr->get_im(input, false);
				im_ptr conversed = app_ptr->converser_ptr->run(from_cs, to_cs, *src);
				app_ptr->put_im(output, *conversed);
			}
			else if (cmd["exe"] == "rotate") { assert_init();
				std::string input = cmd["arg0"], output = cmd["-o"];
				std::string angle = cmd["-a"], algo = cmd["-t"];
				if (angle.empty() || input.empty() || output.empty()) {
					report_cmd("rotate"); continue;
				}
				if (algo.empty()) { algo = "shear"; }
				im_ptr src = app_ptr->get_im(input);
				im_ptr rotated = app_ptr->rotator_ptr->run(algo, atof(angle.c_str()), *src);
				app_ptr->put_im(output, *rotated);
			}
			else if (cmd["exe"] == "gauss") { assert_init();
				std::string input = cmd["arg0"], output = cmd["-o"];
				std::string sigma = cmd["-s"], window = cmd["-w"];
				if (input.empty() || output.empty()) {
					report_cmd("gauss"); continue;
				}
				float sigma_val = 1.5f; int win_size = 5;
				if (!sigma.empty()) { sigma_val = (float)atof(sigma.c_str()); }
				if (!window.empty()) { win_size = atoi(window.c_str()); }
				im_ptr src = app_ptr->get_im(input);
				im_ptr blured = app_ptr->filter_ptr->gauss(sigma_val, win_size, *src);
				app_ptr->put_im(output, *blured);
			}
			else if (cmd["exe"] == "contrast") { assert_init();
				std::string input = cmd["arg0"], output = cmd["-o"];
				if (input.empty() || output.empty()) { report_cmd("contrast"); continue; }

				contraster::args params(cmd["-t"], cmd["-v"], cmd["-c"], cmd["-e"], cmd["-r"]);
				im_ptr src = app_ptr->get_im(input, false);
				im_ptr contrasted = app_ptr->contraster_ptr->run(*src, params);
				app_ptr->put_im(output, *contrasted);
			}
			else if (cmd["exe"] == "wavelet") { assert_init();
				std::string input = cmd["arg0"], output = cmd["-o"];
				std::string basis = cmd["-b"], thresholding = cmd["-t"];
				if (input.empty() || output.empty()) { report_cmd("wavelet"); continue; }
				if (basis.empty()) { basis = "haar"; }
				if (thresholding.empty()) { thresholding = "soft"; }
				float threshold_val = 0.01f;
				if (!thresholding.empty()) { threshold_val = (float)atof(thresholding.c_str()); }
				im_ptr src = app_ptr->get_im(input);
				im_ptr wave = app_ptr->wavelet_ptr->run(basis, threshold_val, *src);
				app_ptr->put_im(output, *wave);
			}
			else if (cmd["exe"] == "env") {
				if (app_ptr != nullptr) { app_ptr->env_info(); }
				else { hardware::env_info(); }
			}
			else if (cmd["exe"] == "dev") { assert_init(); 
				hardware::device_info(app_ptr->env.cur_device);
			}
			else if (cmd["exe"] == "init") {
				std::string platform = cmd["-p"], device = cmd["-d"];
				if (platform.empty() || device.empty()) { platform = cmd["arg0"], device = cmd["arg1"];
					if (platform.empty() || device.empty()) { report_cmd("init"); continue; }
				}
				delete app_ptr;
				app_ptr = new app(atoi(platform.c_str()), atoi(device.c_str()));
			}
			else if (cmd["exe"] == "quit") { break; }
			else { std::cout << "No such command: " << cmd["exe"] << std::endl; }
		}
		catch (std::runtime_error e) { std::cerr << e.what() << std::endl; }
	}
	delete app_ptr;
}
