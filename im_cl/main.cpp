#include"app.h"

app* app_ptr = nullptr;

enum class commands { INIT, ENV, DEV, QUIT, ZOOM, CONVERSE, ROTATE, CONTRAST, GAUSS, WAVELET };

std::unordered_map<std::string, commands> command_ids = {
		{"init", commands::INIT}, {"env", commands::ENV}, {"dev", commands::DEV}, {"quit", commands::QUIT},
		{"zoom", commands::ZOOM}, {"converse", commands::CONVERSE}, {"rotate", commands::ROTATE},
	{"contrast", commands::CONTRAST}, {"gauss", commands::GAUSS}
};

std::unordered_map<commands, std::string> cmd_syntax = {
	{commands::ZOOM, "zoom [-i] <input> -o <output> [-t <type>] ([-f <factor>] | [-x <x> -y <y>])"},
	{commands::INIT, "init [-p] <platform_id> [-d] <device_id> [-m storage_size]"},
	{commands::CONVERSE, "converse [-i] <input> -o <output> [-t <to_cs>] [-f <from_cs>]"},
	{commands::ROTATE, "rotate [-i] <input> -o <output> -a <angle> [-x <center.x> -y <center.y>] [-t <type>]"},
	{commands::GAUSS, "gauss <input> -o <output> [-s <sigma>] [-w <window_size>]"},
	{commands::CONTRAST, "contrast <input> -o <output> [-t <type>] [-v <via_space>] [-c <contrast_val>] [-e <exclusion>] [-x <x_region> -y <y_region>] "},
	{commands::WAVELET, "wavelet <input> -o <output> [-b <basis>] [-t <threshold>]"}
};

struct wrong_usage : public std::runtime_error {
	wrong_usage() : std::runtime_error("Wrong usage, expected:\n") {}
};

void assert_init() {
	if (app_ptr != nullptr) { return; }
	throw std::runtime_error("Not initialised");
}


int main(int argc, char** argv) {
	try { app_ptr = new app(0, 0); }
	catch (std::runtime_error e) { 
		std::cerr << " Default init failed:" << 
			std::endl << e.what() << std::endl;
	}
	while (true) {
		std::cout << "> ";
		command cmd = util::next_action();
		auto cmd_index = command_ids.find(cmd.first);
		if (cmd_index == command_ids.end()) {
			std::cerr << "Unknown command: " << cmd.first << std::endl;
			continue;
		}
		try {
			switch (cmd_index->second) {
			case commands::INIT: {
				std::string platform = cmd.second["-p"];
				if (platform.empty()) { platform = cmd.second["arg0"]; }
				std::string device = cmd.second["-d"];
				if (platform.empty()) { device = cmd.second["arg1"]; }
				if (platform.empty() || device.empty()) { throw wrong_usage(); }

				delete app_ptr;
				app_ptr = new app(atoi(platform.c_str()), atoi(device.c_str()));
				break;
			}
			case commands::ENV: {
				if (app_ptr != nullptr) { app_ptr->env_info(); }
				else { hardware::env_info(); }
				break;
			}
			case commands::DEV: {
				assert_init();
				hardware::device_info(app_ptr->env.cur_device);
				break;
			}
			case commands::QUIT: { goto app_exit; }

			case commands::ZOOM: {
				assert_init();
				std::string input = cmd.second["-i"], kern_type = cmd.second["-t"];
				if (input.empty()) { input = cmd.second["arg0"]; }
				if (kern_type.empty()) { kern_type = "bilinear"; }
				if (input.empty() || cmd.second["-o"].empty()) { throw wrong_usage(); }

				im_ptr src = app_ptr->get_im(input);
				im_ptr scaled = nullptr;

				if (cmd.second["-x"].empty() && cmd.second["-y"].empty()) {
					if (cmd.second["-f"].empty()) { throw wrong_usage(); }
					float factor = atoi(cmd.second["-f"].c_str()) / 100.0f;
					scaled = app_ptr->zoomer_ptr->run(kern_type, factor, src);
				}
				else {
					if (cmd.second["-x"].empty() || cmd.second["-y"].empty()) { throw wrong_usage(); };
					int new_x = atoi(cmd.second["-x"].c_str()), new_y = atoi(cmd.second["-y"].c_str());
					scaled = app_ptr->zoomer_ptr->precise(src, { new_x, new_y });
				}
				app_ptr->put_im(cmd.second["-o"], scaled);
				break;
			}
			case commands::CONVERSE: {
				assert_init();
				std::string input = cmd.second["-i"];
				std::string to = cmd.second["-t"], from = cmd.second["-f"];
				if (input.empty()) { input = cmd.second["arg0"]; }

				if (to.empty() && from.empty()) { throw wrong_usage(); }
				if (input.empty() || cmd.second["-o"].empty()) { throw wrong_usage(); }

				if (from.empty()) { from = "srgb"; }
				else if (to.empty()) { to = "srgb"; }

				im_ptr src = app_ptr->get_im(input, GAMMA_CORRECTION_OFF);
				im_ptr conversed = app_ptr->converser_ptr->run({ from, to }, src);
				app_ptr->put_im(cmd.second["-o"], conversed, GAMMA_CORRECTION_OFF);
				break;
			}
			case commands::ROTATE: {
				assert_init();
				std::string input = cmd.second["-i"], algo = cmd.second["-t"];
				if (input.empty()) { input = cmd.second["arg0"]; }
				if (cmd.second["-a"].empty() || input.empty() ||
					cmd.second["-o"].empty()) { throw wrong_usage(); }
				
				if (algo.empty()) { algo = "shear"; }
				im_ptr src = app_ptr->get_im(input);
				im_ptr rotated = nullptr;
				if (algo == "clockwise" || algo == "counter_clockwise") {
					rotated = app_ptr->rotator_ptr->simple_angle(algo, src);
				}
				else {
					cl_int2 center = { src->size.x / 2, src->size.y / 2 };
					if (!cmd.second["-x"].empty() && !cmd.second["-y"].empty()) {
						center.x = atoi(cmd.second["-x"].c_str());
						center.y = atoi(cmd.second["-y"].c_str());
					}
					double theta = atof(cmd.second["-a"].c_str());
					rotated = app_ptr->rotator_ptr->run(algo, theta, center, src);
				}
				app_ptr->put_im(cmd.second["-o"], rotated);
				break;
			}
			case commands::CONTRAST: {
				assert_init();
				std::string input = cmd.second["arg0"], algo = cmd.second["-t"];
				if (input.empty()) { input = cmd.second["-i"]; }
				if (input.empty() || cmd.second["-o"].empty()) { throw wrong_usage(); }
				im_ptr src = app_ptr->get_im(input, GAMMA_CORRECTION_OFF);
				int channel_mode = contraster::all_channels;
				if (!cmd.second["-v"].empty()) {
					channel_mode = contraster::single_channel;
					im_ptr coloured = app_ptr->converser_ptr->run({ "srgb", cmd.second["-v"] }, src);
					src.swap(coloured);
				}
				if (algo.empty()) { algo = "manual"; }
				im_ptr contrasted = nullptr;
				if (algo == "manual") {
					if (cmd.second["-c"].empty()) { throw wrong_usage(); }
					float c_val = static_cast<float>(atof(cmd.second["-c"].c_str()));
					contrasted = app_ptr->contraster_ptr->manual(src, c_val, channel_mode);
				}
				else if (algo == "exclusive") {
					std::string excl_str = cmd.second["-e"];
					if (excl_str.empty()) { excl_str = "0.39"; }
					float exclusion = static_cast<float>(atof(excl_str.c_str())) / 100.0f;
					contrasted = app_ptr->contraster_ptr->exclusive_hist(src, exclusion, channel_mode);
				}
				else if (algo == "adaptive") {
					if (cmd.second["-x"].empty() || cmd.second["-y"].empty() ||
						cmd.second["-e"].empty()) { throw wrong_usage(); }
					cl_int2 region = { atoi(cmd.second["-x"].c_str()), atoi(cmd.second["-y"].c_str()) };
					int exclude = atoi(cmd.second["-e"].c_str());
					contrasted = app_ptr->contraster_ptr->adaptive_hist(src, region, exclude, channel_mode);
				}
				else { throw std::runtime_error("Unknown contrast: " + algo); }
				if (!cmd.second["-v"].empty()) {
					channel_mode = contraster::single_channel;
					im_ptr decoloured = app_ptr->converser_ptr->run({ cmd.second["-v"], "srgb" }, src);
					contrasted.swap(decoloured);
				}
				app_ptr->put_im(cmd.second["-o"], contrasted, GAMMA_CORRECTION_OFF);
				break;
			}
			case commands::GAUSS: {
				std::string input = cmd.second["arg0"];
				if (input.empty()) { input = cmd.second["-i"]; }
				if (input.empty() || cmd.second["-o"].empty()) { throw wrong_usage(); }
				float sigma_val = 1.0f; int win_size = 3;
				if (!cmd.second["-s"].empty()) { sigma_val = (float)atof(cmd.second["-s"].c_str()); }
				if (!cmd.second["-w"].empty()) { win_size = atoi(cmd.second["-w"].c_str()); }
				im_ptr src = app_ptr->get_im(input);
				im_ptr blured = app_ptr->filter_ptr->gauss(sigma_val, win_size, src);
				app_ptr->put_im(cmd.second["-o"], blured);
				break;
			}
			}

			/*if (cmd["exe"] == "zoom") { assert_init();
				std::string input = cmd["arg0"], output = cmd["-o"];
				std::string scale = cmd["-f"], kern_type = cmd["-t"];
				if (scale.empty() || input.empty() || output.empty()) { 
					report_cmd("zoom"); continue;
				}
				float factor = atoi(scale.c_str()) / 100.0f;
				if (kern_type.empty()) { kern_type = "lan3"; }
				im_ptr src = app_ptr->get_im(input);
				im_ptr scaled = app_ptr->zoomer_ptr->run(kern_type, factor, *src);
				app_ptr->put_im(output, scaled);
			}*/
			/*else if (cmd["exe"] == "converse") { assert_init();
				std::string input = cmd["arg0"], output = cmd["-o"];
				std::string to_cs = cmd["-t"], from_cs = cmd["-f"];
				if ((to_cs.empty() && from_cs.empty()) || input.empty() || output.empty()) {
					report_cmd("converse"); continue;
				}
				if (from_cs.empty()) { from_cs = "srgb"; }
				else if (to_cs.empty()) { to_cs = "srgb"; }
				im_ptr src = app_ptr->get_im(input, false);
				im_ptr conversed = app_ptr->converser_ptr->run(from_cs, to_cs, *src);
				app_ptr->put_im(output, conversed, false);
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
				app_ptr->put_im(output, rotated);
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
				app_ptr->put_im(output, blured);
			}
			else if (cmd["exe"] == "contrast") { assert_init();
				std::string input = cmd["arg0"], output = cmd["-o"];
				if (input.empty() || output.empty()) { report_cmd("contrast"); continue; }

				contraster::args params(cmd["-t"], cmd["-v"], cmd["-c"], cmd["-e"], cmd["-r"]);
				im_ptr src = app_ptr->get_im(input, false);
				im_ptr contrasted = app_ptr->contraster_ptr->run(*src, params);
				app_ptr->put_im(output, contrasted);
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
				app_ptr->put_im(output, wave);
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
			else { std::cout << "No such command: " << cmd["exe"] << std::endl; }*/
		}
		catch (wrong_usage e) { std::cerr << e.what() << cmd_syntax[cmd_index->second]; }
		catch (std::runtime_error e) { std::cerr << e.what() << std::endl; }
	}
	app_exit:
	delete app_ptr;
}
