#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <fstream>
#include <iomanip>

#include "markerdetection.h"
int main(int argc, char* argv[]) {
	// Check for parameters
	if (argc == 1) {
		std::cerr << "Usage: marker_detection <image folder path> [show detected markers (0/1)]" << std::endl;
		exit(1);
	}

	// Read image path
	std::string folderpath = argv[1];

	// Visualization
	int show_markers = 0;
	if (argc == 3) {
		show_markers = (std::atoi(argv[2]));
	}

	// Open output files
	std::ofstream resultfile_param(folderpath + "/image_points.txt");
	std::ofstream stats_file(folderpath + "/stats.txt");

	if (!stats_file.is_open() || !resultfile_param.is_open()) {
		std::cerr << "Error: Unable to write to " << folderpath << std::endl;
		exit(1);
	}
	resultfile_param << "image_name point_id x y a b angle mdan" << std::endl;
	stats_file << "image_name found_markers coded_markers time_needed_ms" << std::endl;

	// Get file names in directory 
	std::vector<cv::String> filenames;
	try {
		cv::glob(folderpath, filenames);
	}
	catch (const cv::Exception& e) {
		std::cerr << e.msg << std::endl;
		exit(1);
	}

	// Iteration through files
	cv::Mat image, image_gray;
	int n_markers = 0;
	int n_images = 0;
	double time_elapsed = 0;
	for (auto const& file_path : filenames) {
		image = cv::imread(file_path);
		if (image.empty()) {
			continue;
		}
		n_images++;

		// Convert to grayscale
		if (image.channels() == 3) {
			cv::cvtColor(image, image_gray, cv::COLOR_BGR2GRAY);
		}
		else {
			image_gray = image;
		}

		// Find markers
		double t = (double)cv::getTickCount();
		std::vector<marker_detection::Ellipse> detectedmarkers;
		marker_detection::detectAndDecode(image_gray, detectedmarkers, marker_detection::Parameter());
		t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
		time_elapsed += t;

		// Generating output files
		int n_coded_markers = 0;
		std::string file_name = file_path.substr(file_path.length() - (file_path.length() - folderpath.length() - 1));
		for (auto const& m : detectedmarkers) {
			resultfile_param << std::setprecision(4) << std::fixed
				<< file_name << " "
				<< m.id << " "
				<< m.x << " "
				<< m.y << " "
				<< m.a << " "
				<< m.b << " "
				<< m.angle << " "
				<< m.mdan << std::endl;
			if (m.id != -1) {
				n_coded_markers++;
			}
		}

		stats_file << file_name << " "
			<< detectedmarkers.size() << " "
			<< n_coded_markers << " "
			<< t * 1000 << std::endl;

		n_markers += n_coded_markers;

		if (show_markers > 0) {
			int thickness = (image.cols > 1000) ? image.cols / 1000.0 : 1;
			// Draw
			for (auto& m : detectedmarkers) {
				cv::ellipse(image, cv::Point2f(m.x, m.y), cv::Size2f(m.b, m.a), m.angle / CV_PI * 180.0, 0, 360, cv::Scalar(0, 255, 255), thickness);
				if (m.id > 0) {
					std::string label = std::to_string(m.id);
					cv::putText(image, label, cv::Point(m.x, m.y), cv::FONT_HERSHEY_PLAIN, thickness, cv::Scalar(0, 0, 255), thickness);
				}

				// Drawing angle of ellipse
				cv::Point2f point2 = cv::Point2f(
					m.a * cos(m.angle - CV_PI * 0.5) * 2.0 + m.x,
					m.a * sin(m.angle - CV_PI * 0.5) * 2.0 + m.y);
				cv::line(image, m.point(), point2, cv::Scalar(0, 0, 255), thickness);
			}

			for (auto& m : detectedmarkers) {
				if (m.debug) {
					for (int i = 0; i < m.debug->edge_points_sub_pixel.size(); i++) {
						cv::drawMarker(image, m.debug->edge_points_sub_pixel[i], cv::Scalar(255,255,255), 0, 1, 1);
					}

					for (auto& p : m.debug->edge_points_robust)
						cv::drawMarker(image, p, cv::Scalar(255, 0, 0), 0, 5, 1);
				}
			}

			// Show
			cv::namedWindow("Detected Markers", cv::WINDOW_GUI_EXPANDED);
			cv::imshow("Detected Markers", image);
			cv::waitKey();

			// Write
			// cv::imwrite(folderpath + "/" + file_name + "_out.jpg", image);
		}
	}
	std::cerr << time_elapsed << " seconds to detect " << n_markers << " coded markers in " << n_images << " images (" << (time_elapsed / double(n_images)) * 1000 << " ms per image)." << std::endl;

	resultfile_param.close();
	stats_file.close();

	exit(0);
}
