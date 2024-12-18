#include "markerdetection.h"

#include <map>
#include <iostream>
#include <numeric>
#include "codes.h"

namespace marker_detection {

	void detectAndDecode(const cv::Mat& image, std::vector<Ellipse>& markers, Parameter param, DetectionDebug* debug)
	{
		if (image.empty()) {
			std::cerr << "Error: empty image!" << std::endl;
			return;
		}
		if (image.type() != CV_8U) {
			std::cerr << "Error: only for images of type CV_8U!" << std::endl;
			return;
		}

		// Detect edge points of marker in image
		std::vector<std::vector<cv::Point>> edgePoints;
		findConnectedEdgePoints(image, edgePoints, param, debug);
		unsigned int minEdgePixel = int(round(CV_PI * param.marker_min_diameter)); // circumference = diameter*PI (assuming circle)
		unsigned int maxEdgepixel = int(round(CV_PI * param.marker_max_diameter));
		Ellipse ellipse;
		std::vector<Ellipse> temp_uncodedmarkers;
		std::vector<int> found_ids;
		for (auto const& edgePixel : edgePoints)
		{
			// Ignore large and small contours
			if (edgePixel.size() < minEdgePixel ||
				edgePixel.size() > maxEdgepixel) {
				continue;
			}

			// Initial ellipse fit
			Ellipse ellipse;
			fitEllipse(edgePixel, ellipse, param.ellipse_fit_type);
			// fitEllipseRobust() at this point very expensive

			// Small ellipse
			if (ellipse.a * 2.0 < param.marker_min_diameter || ellipse.b * 2.0 < param.marker_min_diameter) {
				continue;
			}

			// Ratio
			if (ellipse.a / ellipse.b > param.max_ellipse_ratio) {
				continue;
			}

			// Get mean pixel value of marker and of surrounding area
			double marker_value, marker_rmse, surrounding_value, surrounding_rmse;
			checkMarkerSurrounding(ellipse, image, marker_value, marker_rmse, surrounding_value, surrounding_rmse);

			// Contrast between marker and surroundings and rmse
			if (abs(marker_value - surrounding_value) < param.marker_min_contrast ||
				marker_rmse > param.max_marker_value_rmse ||
				surrounding_rmse > param.max_surrounding_value_rmse) {
				continue;
			}

			// Check if white marker on black background
			bool marker_white = (marker_value > surrounding_value);
			double middle_value = (marker_value + surrounding_value) * 0.5;

			// Activate debug
			if (debug) {
				ellipse.debug = std::make_shared<EllipseDebug>();
				ellipse.debug->edge_points_initial = edgePixel;
				ellipse.debug->marker_value = marker_value;
				ellipse.debug->marker_rmse = marker_rmse;
				ellipse.debug->surrounding_value = surrounding_value;
				ellipse.debug->surrounding_rmse = surrounding_rmse;
			}

			if (param.sub_pixel_method > 0) { // Optional sub pixel measurement
				if (!subPixelMeasurement(image, ellipse, param, marker_value, surrounding_value))
					continue;
			}
			else if (param.robust_ellipse_fit) { // If no sub pixel measurement and robust fit
				if (!fitEllipseRobust(ellipse, edgePixel, param.ellipse_fit_type))
					continue;
			}

			// Decode marker
			if (param.detect_coded_marker) {
				if (decodeMarker(image, ellipse, param, middle_value, marker_white)) {
					// TODO: remove and later check if some ID´s are detected more than once.
					if (std::find(found_ids.begin(), found_ids.end(), ellipse.id) == found_ids.end()) {
						markers.push_back(ellipse);
						found_ids.push_back(ellipse.id);
						continue;
					}
				}
			}

			// Save as uncoded marker
			if (param.return_uncoded_marker) {
				temp_uncodedmarkers.push_back(ellipse);
			}
		}

		// Add uncoded markers to markers if not close to other coded marker
		// TODO: allow to skip this check
		for (auto const& ucm : temp_uncodedmarkers) {
			bool skip = false;
			for (auto const& cm : markers) {
				double lim = (cm.id < 0) ? cm.a * param.min_distance_closest_point : cm.a * 1.2 * param.code_ring_radius * param.min_distance_closest_point;
				if (abs(ucm.x - cm.x) < lim && abs(ucm.y - cm.y) < lim)
					skip = true;
			}

			if (!skip) {
				markers.push_back(ucm);
			}
		}
	}

	float getSubPixValue(const cv::Mat& image, const cv::Point2f& point)
	{
		// Bilinear interpolation
		int x0 = static_cast<int>(std::floor(point.x));
		int y0 = static_cast<int>(std::floor(point.y));

		float a = point.x - x0;
		float c = point.y - y0;

		// Store intermediate results to avoid redundant memory access
		const uchar* image_ptr = image.ptr<uchar>(y0);
		const uchar* image_ptr_next = image.ptr<uchar>(y0 + 1);
		float top_interp = image_ptr[x0] * (1.f - a) + image_ptr[x0 + 1] * a;
		float bottom_interp = image_ptr_next[x0] * (1.f - a) + image_ptr_next[x0 + 1] * a;

		// Final interpolation
		return top_interp * (1.f - c) + bottom_interp * c;
	}

	std::optional<float> getSubPixValueWithChecks(const cv::Mat& image, const cv::Point2f& point)
	{
		// Check if point outside of image
		if (point.x < 0 || point.x >= image.cols - 1 ||
			point.y < 0 || point.y >= image.rows - 1) {
			return std::nullopt;
		}

		return getSubPixValue(image, point);
	}

	template<typename T>
	void momentPreservation(const std::vector<T>& values, float& pos, float& h1, float& h2)
	{
		pos = h1 = h2 = 0.0;

		if (values.size() < 5)
			return;

		// Accumulate sums for moments
		double sum = 0, sum2 = 0, sum3 = 0;
		T last = values[0];
		T step = T(0);
		for (auto const& v : values) {
			sum += v;
			sum2 += v * v;
			sum3 += v * v * v;

			step += (v - last);
			last = v;
		}

		// First, second, and third moments
		double size = double(values.size());
		double m1 = sum / size;
		double m2 = sum2 / size;
		double m3 = sum3 / size;
		double sigma = sqrt(m2 - m1 * m1);

		// Retrun early if sigma < 0
		if (sigma <= std::numeric_limits<double>::epsilon()) {
			return;
		}

		double s = (m3 + 2.0 * (m1 * m1 * m1) - 3 * m1 * m2) / (sigma * sigma * sigma);
		double p1 = (1.0 + s * sqrt(1.0 / (4.0 + s * s))) / 2.0;
		double p2 = 1.0 - p1;
		pos = size * p1 - 0.5;
		double pos2 = size * p2 - 0.5;
		double hh1 = m1 - sigma * sqrt(p2 / p1);
		double hh2 = m1 + sigma * sqrt(p1 / p2);

		if (step < T(0)) {
			pos = pos2;
			h1 = hh2;
			h2 = hh1;
		}
		else {
			h1 = hh1;
			h2 = hh2;
		}
	}
	template void momentPreservation<int>(const std::vector<int>& values, float& x, float& min, float& max);
	template void momentPreservation<float>(const std::vector<float>& values, float& x, float& min, float& max);
	template void momentPreservation<double>(const std::vector<double>& values, float& x, float& min, float& max);

	void fitEllipse(cv::InputArray& edge_points, Ellipse& ellipse, int type)
	{
		cv::RotatedRect box;
		switch (type) {
		case 1:
			box = cv::fitEllipseAMS(edge_points);
			break;
		case 2:
			box = cv::fitEllipseDirect(edge_points);
			break;
		default:
			box = cv::fitEllipse(edge_points);
		}

		// Ellipse (box.height allways larger than box.width)
		ellipse.x = box.center.x;
		ellipse.y = box.center.y;
		ellipse.a = box.size.height / 2.0;
		ellipse.b = box.size.width / 2.0;
		ellipse.angle = box.angle * CV_PI / 180.0;
	}

	bool fitEllipseRobust(Ellipse& ellipse, cv::InputArray& edge_points,
		int type, int max_iterations, double threshold) {
		// see:  https://doi.org/10.1016/j.isprsjprs.2021.04.010
		// TODO: allow to limit the number of points used (n_points > 100)

		// Fit initial ellipse if not computed
		if (!ellipse.isValid())
			fitEllipse(edge_points, ellipse, type);

		cv::Mat _points = edge_points.getMat();
		const cv::Point* ptsi = _points.ptr<cv::Point>();
		const cv::Point2f* ptsf = _points.ptr<cv::Point2f>();
		bool isFloat = edge_points.depth() == CV_32F;
		int n_points = _points.size().width;
		int n_last_inliers = n_points;

		for (int itt = 0; itt < max_iterations; itt++) {
			// Transform points to a circle and calculate distance to center
			/*
				cv::Mat transform_ellipse = (cv::Mat_<double>(2, 2) << 1.0 / ellipse.b, 0, 0, 1.0 / ellipse.a) *
					(cv::Mat_<double>(2, 2) << cos(ellipse.angle), sin(ellipse.angle), -sin(ellipse.angle), cos(ellipse.angle));
				for (int j = 0; j < n_points; j++) {
					cv::Point2f p = isFloat ? ptsf[j] : cv::Point2f((float)ptsi[j].x, (float)ptsi[j].y);
					cv::Mat cp = transform_ellipse * (cv::Mat_<double>(2, 1) << p.x - ellipse.x, p.y - ellipse.y);
				}
			*/

			// Precompute values for transformation
			double cos_angle = cos(ellipse.angle);
			double sin_angle = sin(ellipse.angle);
			double inv_a = 1.0 / ellipse.a;
			double inv_b = 1.0 / ellipse.b;

			std::vector<double> dist(n_points);
			double dx, dy;
			for (int j = 0; j < n_points; j++) {
				if (isFloat) {
					dx = ptsf[j].x - ellipse.x;
					dy = ptsf[j].y - ellipse.y;
				}
				else {
					dx = static_cast<float>(ptsi[j].x) - ellipse.x;
					dy = static_cast<float>(ptsi[j].y) - ellipse.y;
				}

				// Transform the point
				double transformed_x = inv_b * (cos_angle * dx + sin_angle * dy);
				double transformed_y = inv_a * (-sin_angle * dx + cos_angle * dy);

				// Compute the Euclidean norm
				dist[j] = std::sqrt(transformed_x * transformed_x + transformed_y * transformed_y);
			}

			// Calculate median of distances, the differences for each point to the median and the median of the differences
			double median = marker_detection::calculateMedian(dist);
			std::vector<double> diff(n_points);
			std::transform(dist.begin(), dist.end(), diff.begin(),
				[median](double d) { return std::abs(d - median); });
			double madn = marker_detection::calculateMedian(diff) / 0.67499;
			double mod = estimateModeKDE(dist);

			// Extract inlier points based on threshold
			std::vector<cv::Point2f> inl_points;
			for (int j = 0; j < n_points; j++) {
				if (abs(dist[j] - mod) / madn <= threshold) {
					inl_points.push_back(isFloat ? ptsf[j] : cv::Point2f((float)ptsi[j].x, (float)ptsi[j].y));
				}
			}

			// Break if not enough inlier points
			if (inl_points.size() < 5)
				break;

			if (ellipse.debug) {
				ellipse.debug->edge_points_robust = inl_points;
			}

			// Break if number of point did not change
			ellipse.mdan = madn;
			if (inl_points.size() == n_last_inliers) {
				break;
			}
			n_last_inliers = inl_points.size();
			fitEllipse(inl_points, ellipse, type);
		}
		return true;
	}

	bool starOperator(const cv::Mat& image, Ellipse& ellipse,
		Parameter& param, const double& marker_value, const double& marker_around)
	{
		float min_radius = 0.5;
		float max_radius = 1.5;

		int scan_line_steps = ceil(ellipse.a * 2.0); //~0,5 pixel
		if (scan_line_steps < 5) {
			scan_line_steps = 5;
		}

		double scan_angle_step = CV_2PI / double(param.sub_pixel_scan_lines);
		double min_contrast = abs(marker_value - marker_around) * param.marker_contrast_consistency;
		// Scanning around marker
		for (int itt = 0; itt < param.sub_pixel_iterations; itt++) {
			std::vector<cv::Point2f> edgePoints;
			double alpha = sin(ellipse.angle - PI_HALF);
			double beta = cos(ellipse.angle - PI_HALF);
			for (unsigned int i = 0; i < param.sub_pixel_scan_lines; i++) {
				double scan_angle = double(i) * scan_angle_step;
				float xPos = ellipse.a * cos(scan_angle);
				float yPos = ellipse.b * sin(scan_angle);

				// Extract pixel values on scanline
				std::vector<float> scanline;
				sampleScanlineEllipse(image, ellipse.x, ellipse.y, scanline, xPos, yPos, alpha, beta, scan_line_steps, min_radius, max_radius);

				float pos, h1, h2;
				momentPreservation(scanline, pos, h1, h2);

				if (abs(h1 - h2) > min_contrast && pos > 0) {
					double radius = min_radius + 1.0 / double(scanline.size()) * (pos); // -1???
					float x = ellipse.x + xPos * beta * radius - yPos * alpha * radius;
					float y = ellipse.y + xPos * alpha * radius + yPos * beta * radius;
					edgePoints.emplace_back(x, y);
				}
			}

			if (edgePoints.size() < (double(param.sub_pixel_scan_lines) * 0.7) || edgePoints.size() < 4) {
				return false;
			}

			marker_detection::fitEllipse(edgePoints, ellipse, param.ellipse_fit_type);

			if (ellipse.debug)
				ellipse.debug->edge_points_sub_pixel = edgePoints;

			// Robust ellipse fit
			if (param.robust_ellipse_fit) {
				if (!fitEllipseRobust(ellipse, edgePoints, param.ellipse_fit_type))
					return false;
			}
		}
		return true;
	}


	void findConnectedEdgePoints(const cv::Mat& image, std::vector<std::vector<cv::Point> >& markers,
		Parameter& param, DetectionDebug* debug)
	{
		// Bluring image
		cv::Mat _image;
		if (param.median_blur_kernel > 2) {
			adjustKernelSize(param.median_blur_kernel);
			cv::medianBlur(image, _image, param.median_blur_kernel);
		}
		if (param.blur_kernel > 2) {
			adjustKernelSize(param.blur_kernel);
			cv::blur(image, _image, cv::Size(param.blur_kernel, param.blur_kernel));
		}
		if (_image.empty())
			_image = image;

		// Threshold image
		cv::Mat threshold;
		applyThreshold(_image, threshold, param);

		// Find edge
		cv::Mat edge;
		if (param.edge_method == 1) {
			adjustKernelSize(param.adaptive_threshold_block_size);
			if (param.adaptive_threshold_block_size < 3) param.adaptive_threshold_block_size = 7;
			cv::adaptiveThreshold(threshold, edge, 255, param.adaptive_threshold_method, cv::THRESH_BINARY, param.adaptive_threshold_block_size, param.adaptive_threshold_C);

			// Invert if markers are black and background is white
			if (cv::mean(edge)[0] > 127)
				cv::bitwise_not(edge, edge);

			// Thinning of the edges
			// copy of https://github.com/opencv/opencv_contrib/blob/4.x/modules/ximgproc/src/thinning.cpp
			int  thinningType = 0;
			cv::Mat processed = edge.clone();
			CV_CheckTypeEQ(processed.type(), CV_8UC1, "");
			processed /= 255;
			cv::Mat prev = cv::Mat::zeros(processed.size(), CV_8UC1);
			cv::Mat diff;
			do {
				thinningIteration(processed, 0, thinningType);
				thinningIteration(processed, 1, thinningType);
				absdiff(processed, prev, diff);
				processed.copyTo(prev);
			} while (countNonZero(diff) > 0);

			processed *= 255;
			edge = processed;
		}
		else { // Default
			cv::Canny(threshold, edge, param.canny_threshold1, param.canny_threshold2);
		}

		// Extract connected components
		cv::Mat lable;
		int nComponents = cv::connectedComponents(edge, lable);
		markers.resize(nComponents);
		for (int r = 0; r < lable.rows; r++) {
			int* ptr = lable.ptr<int>(r);
			for (int c = 0; c < lable.cols; c++) {
				int component = *ptr++;
				if (component > 0)
					markers[component].emplace_back(c, r);
			}
		}

		// Debug
		if (debug) {
			debug->threshold = threshold;
			debug->edge = edge;
			debug->connected_edges = markers;
		}
	}

	bool zhouOperator(const cv::Mat& image, Ellipse& ellipse,
		double marker_value, double marker_around, const Parameter& param)
	{
		// TODO: Check and clean up code
		double bandwidth = 0.75;
		unsigned int numberOfSteps = 120;
		double min_contrast = abs(marker_value - marker_around) * 0.75;

		// vertical: key=y, value=x, horizontal: key=x, value=y
		std::map<int, double> coor_upper, coor_lower, coor_left, coor_right;
		cv::Point2i point = cv::Point2i();
		double scan_angle_step = (CV_2PI) / double(numberOfSteps);

		for (int itt = 0; itt < param.sub_pixel_iterations; itt++) {
			double alpha = sin(ellipse.angle - PI_HALF);
			double beta = cos(ellipse.angle - PI_HALF);

			for (unsigned int a = 0; a < numberOfSteps; a++) {
				// Calculate Point
				double scan_angle = double(a) * scan_angle_step;
				float xPos = ellipse.a * cos(scan_angle);
				float yPos = ellipse.b * sin(scan_angle);
				point.x = round(ellipse.x + xPos * beta - yPos * alpha);
				point.y = round(ellipse.y + xPos * alpha + yPos * beta);

				if (point.x < 0 || point.y < 0 ||
					point.x >= image.cols || point.y >= image.rows)
					return false;

				// -- Row
				int bandwidth_p = round(abs(point.x - ellipse.x) * bandwidth);
				auto mappos = (point.x < ellipse.x) ? &coor_left : &coor_right;
				if (bandwidth_p > 2 && mappos->find(point.y) == mappos->end()) {
					// Check if on edge
					if (point.x - bandwidth_p < 0 || point.x + bandwidth_p >= image.cols)
						return false;

					cv::Mat row = image.row(point.y);
					std::vector<int> values;
					for (int v = point.x - bandwidth_p; v < point.x + bandwidth_p; v++) {
						values.push_back(row.at<uchar>(v));
					}

					float h_pos, h_h1, h_h2;
					momentPreservation(values, h_pos, h_h1, h_h2);

					if (abs(h_h1 - h_h2) > min_contrast) {
						mappos->insert(std::make_pair(point.y, point.x + h_pos - double(bandwidth_p) /*- 1.0*/));
					}
				}

				// -- Col
				bandwidth_p = round(abs(point.y - ellipse.y) * bandwidth);
				mappos = (point.y < ellipse.y) ? &coor_upper : &coor_lower;
				if (bandwidth_p > 2 && mappos->find(point.y) == mappos->end()) {
					if (point.y - bandwidth_p < 0 || point.y + bandwidth_p >= image.rows)
						return false;

					cv::Mat col = image.col(point.x);
					std::vector<int> values;
					for (int v = point.y - bandwidth_p; v < point.y + bandwidth_p; v++) {
						values.push_back(col.at<uchar>(v));
					}

					float v_pos, v_h1, v_h2;
					momentPreservation(values, v_pos, v_h1, v_h2);

					if (abs(v_h1 - v_h2) > min_contrast) {
						mappos->insert(std::make_pair(point.x, point.y + v_pos - double(bandwidth_p)/*- 1.0*/));
					}
				}
			}

			// Calc vertical middle points and fit line
			std::vector<cv::Point2d> middle;
			std::vector<float> v_line, h_line;
			for (auto const& v : coor_left) {
				auto right = coor_right.find(v.first);
				if (right != coor_right.end())
					middle.push_back(cv::Point2d((v.second + right->second) / 2.0, double(v.first)));
			}
			if (middle.size() < 4)
				return false;
			cv::fitLine(middle, v_line, cv::DIST_L2, 0, 0.001, 0.001);

			// Calc horizontal middle points and fit line
			middle.clear();
			for (auto const& v : coor_upper) {
				auto lower = coor_lower.find(v.first);
				if (lower != coor_lower.end())
					middle.push_back(cv::Point2d(double(v.first), (v.second + lower->second) / 2.0));
			}
			if (middle.size() < 4)
				return false;
			cv::fitLine(middle, h_line, cv::DIST_L2, 0, 0.001, 0.001);

			if (h_line.size() < 4 || v_line.size() < 4)
				return false;

			// line intersection
			// https://stackoverflow.com/questions/7446126/opencv-2d-line-intersection-helper-function
			cv::Point2f x = cv::Point2f(h_line[2], h_line[3]) - cv::Point2f(v_line[2], v_line[3]);
			float cross = v_line[0] * h_line[1] - v_line[1] * h_line[0];
			if (abs(cross) > /*EPS*/1e-8) {
				double t1 = (x.x * h_line[1] - x.y * h_line[0]) / cross;
				cv::Point2f center = cv::Point2f(v_line[2], v_line[3]) + cv::Point2f(v_line[0], v_line[1]) * t1;

				if (abs(center.x - ellipse.x) > ellipse.a || abs(center.y - ellipse.y) > ellipse.a)
					return false;

				ellipse.x = center.x;
				ellipse.y = center.y;

				return true;
			}
		}

		return false;
	}


	bool searchMarker(const cv::Mat& image, Ellipse& ell, Parameter& param) {
		// search in a star pattern for a contrast change (> min contrast in param)
		int search_lines = 10;
		std::vector<std::vector<uchar>> pixel_values(search_lines);
		std::vector<cv::Point> edge;
		cv::Point2i point = cv::Point2i();
		cv::Point2f initial_pos(ell.x, ell.y);
		double scan_angle_step = CV_2PI / double(search_lines);

		for (unsigned int i = 0; i < search_lines; i++) {
			double sin_angle = sin(double(i) * scan_angle_step);
			double cos_angle = cos(double(i) * scan_angle_step);
			double search = true;
			for (int j = 1; j < param.marker_max_diameter / 2 && search; j++) {
				point.x = (double(j) * sin_angle) + ell.x;
				point.y = (double(j) * cos_angle) + ell.y;

				if (point.x < 0 || point.x >= image.cols || point.y < 0 || point.y >= image.rows) {
					search = false;
				}
				else {
					uchar v = image.at<uchar>(point);
					pixel_values[i].push_back(v);
					if (pixel_values[i].size() > param.marker_min_diameter) {
						if (abs(pixel_values[i][0] - v) > param.marker_min_contrast) {
							edge.push_back(point);
							search = false;
						}
					}
				}
			}
		}

		if (edge.size() < 6)
			return false;

		fitEllipse(edge, ell, param.ellipse_fit_type);

		// Roubust
		if (param.robust_ellipse_fit) {
			if (!fitEllipseRobust(ell, edge, param.ellipse_fit_type))
				return false;
		}

		double marker_value, marker_rmse, surrounding_value, surrounding_rmse;
		checkMarkerSurrounding(ell, image, marker_value, marker_rmse, surrounding_value, surrounding_rmse);

		// Contrast between marker and surroundings
		if (abs(marker_value - surrounding_value) < param.marker_min_contrast) {
			return false;
		}

		if (!subPixelMeasurement(image, ell, param, marker_value, surrounding_value)) {
			return false;
		}

		// small/large ellipse
		if (ell.a > param.marker_max_diameter || ell.b < param.marker_min_diameter) {
			return false;
		}

		// Ratio
		if (ell.a / ell.b > param.max_ellipse_ratio) {
			return false;
		}

		if (abs(ell.x - initial_pos.x) > ell.a / 2 || abs(ell.y - initial_pos.y) > ell.a / 2) {
			return false;
		}

		return true;
	}

	bool subPixelMeasurement(const cv::Mat& image, Ellipse& ellipse,
		Parameter& param, double marker_value, double marker_around)
	{
		switch (param.sub_pixel_method) {
		case 1:
			return starOperator(image, ellipse, param, marker_value, marker_around);
		case 2:
			return zhouOperator(image, ellipse, marker_value, marker_around, param);
		default:// No sub pixel measurement
			return true;
		}
	}

	void bradley_adaptive_thresholding(const cv::Mat& in, cv::Mat& out)
	{
		// TODO: check and optimize
		out = cv::Mat(in.size(), CV_8U);

		// rows -> height -> y
		int nRows = in.rows;
		// cols -> width -> x
		int nCols = in.cols;

		// create the integral image
		cv::Mat intImage;
		cv::integral(in, intImage);

		int S = MAX(nRows, nCols) / 8;
		double T = 0.15;

		// perform thresholding
		int s2 = S / 2;
		int x1, y1, x2, y2, count, sum;

		int* p_y1, * p_y2;
		const uchar* p_inputMat;
		uchar* p_outputMat;

		for (int i = 0; i < nRows; ++i)
		{
			y1 = i - s2;
			y2 = i + s2;

			if (y1 < 1)
			{
				y1 = 1;
			}
			if (y2 >= nRows)
			{
				y2 = nRows - 1;
			}
			// p_y1 = intImage.ptr<int>(y1);
			p_y1 = intImage.ptr<int>(y1 - 1);
			p_y2 = intImage.ptr<int>(y2);
			p_inputMat = in.ptr<uchar>(i);
			p_outputMat = out.ptr<uchar>(i);

			for (int j = 0; j < nCols; ++j)
			{
				// set the SxS region
				x1 = j - s2;
				x2 = j + s2;

				if (x1 < 1)
				{
					x1 = 1;
				}
				if (x2 >= nCols)
				{
					x2 = nCols - 1;
				}

				count = (x2 - x1) * (y2 - y1);

				// sum = p_y2[x2] - p_y1[x2] - p_y2[x1] + p_y1[x1];
				sum = p_y2[x2] - p_y1[x2] - p_y2[x1 - 1] + p_y1[x1 - 1];

				if (p_inputMat[j] * count < sum * (1.0 - T))
					p_outputMat[j] = 0;
				else
					p_outputMat[j] = 255;
			}
		}
	}

	void checkMarkerSurrounding(const Ellipse& e, const cv::Mat& image,
		double& marker_value, double& marker_rmse,
		double& surrounding_value, double& surrounding_rmse)
	{
		marker_value = 0;
		marker_rmse = -1;
		surrounding_value = 0;
		surrounding_rmse = -1;
		double alpha = sin(e.angle - PI_HALF);
		double beta = cos(e.angle - PI_HALF);
		std::vector<double> mar, out;
		double a = CV_2PI / 12.0;
		for (int i = 0; i < 12; i++) {
			double x = e.a * cos(double(i) * a);
			double y = e.b * sin(double(i) * a);

			std::vector<float> scanline;
			sampleScanlineEllipse(image, e.x, e.y, scanline, x, y, alpha, beta, 4, 0.5, 1.5);
			for (int j = 0; j < scanline.size(); j++) {
				if (j < 2)
					mar.push_back(scanline[j]);
				else
					out.push_back(scanline[j]);
			}
		}
		if (mar.size() < 12 || out.size() < 12)
			return;

		marker_value = calculateMedian(mar);
		marker_rmse = calculateRMSE(mar);
		surrounding_value = calculateMedian(out);
		surrounding_rmse = calculateRMSE(out);
	}

	// Applies a thinning iteration to a binary image
	// copied: https://github.com/opencv/opencv_contrib/blob/4.x/modules/ximgproc/src/thinning.cpp
	static void thinningIteration(cv::Mat img, int iter, int thinningType) {
		cv::Mat marker = cv::Mat::zeros(img.size(), CV_8UC1);

		if (thinningType == 0) { // THINNING_ZHANGSUEN
			for (int i = 1; i < img.rows - 1; i++)
			{
				for (int j = 1; j < img.cols - 1; j++)
				{
					uchar p2 = img.at<uchar>(i - 1, j);
					uchar p3 = img.at<uchar>(i - 1, j + 1);
					uchar p4 = img.at<uchar>(i, j + 1);
					uchar p5 = img.at<uchar>(i + 1, j + 1);
					uchar p6 = img.at<uchar>(i + 1, j);
					uchar p7 = img.at<uchar>(i + 1, j - 1);
					uchar p8 = img.at<uchar>(i, j - 1);
					uchar p9 = img.at<uchar>(i - 1, j - 1);

					int A = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
						(p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
						(p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
						(p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
					int B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
					int m1 = iter == 0 ? (p2 * p4 * p6) : (p2 * p4 * p8);
					int m2 = iter == 0 ? (p4 * p6 * p8) : (p2 * p6 * p8);

					if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)
						marker.at<uchar>(i, j) = 1;
				}
			}
		}
		if (thinningType == 1) { // THINNING_GUOHALL
			for (int i = 1; i < img.rows - 1; i++)
			{
				for (int j = 1; j < img.cols - 1; j++)
				{
					uchar p2 = img.at<uchar>(i - 1, j);
					uchar p3 = img.at<uchar>(i - 1, j + 1);
					uchar p4 = img.at<uchar>(i, j + 1);
					uchar p5 = img.at<uchar>(i + 1, j + 1);
					uchar p6 = img.at<uchar>(i + 1, j);
					uchar p7 = img.at<uchar>(i + 1, j - 1);
					uchar p8 = img.at<uchar>(i, j - 1);
					uchar p9 = img.at<uchar>(i - 1, j - 1);

					int C = ((!p2) & (p3 | p4)) + ((!p4) & (p5 | p6)) +
						((!p6) & (p7 | p8)) + ((!p8) & (p9 | p2));
					int N1 = (p9 | p2) + (p3 | p4) + (p5 | p6) + (p7 | p8);
					int N2 = (p2 | p3) + (p4 | p5) + (p6 | p7) + (p8 | p9);
					int N = N1 < N2 ? N1 : N2;
					int m = iter == 0 ? ((p6 | p7 | (!p9)) & p8) : ((p2 | p3 | (!p5)) & p4);

					if ((C == 1) && ((N >= 2) && ((N <= 3)) && (m == 0)))
						marker.at<uchar>(i, j) = 1;
				}
			}
		}

		img &= ~marker;
	}

	double calculateMedian(std::vector<double> values)
	{
		unsigned int size = values.size();
		double median = 0.0;
		if (size > 2) {
			std::sort(values.begin(), values.end());
			median = size % 2 == 0 ? (values[size / 2 - 1] + values[size / 2]) / 2 : values[size / 2];
		}
		return median;
	}

	double calculateRMSE(std::vector<double>& values)
	{
		if (values.empty())
			return 0.0;

		double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
		double sum_squared_diff = 0.0;

		for (const double& value : values) {
			sum_squared_diff += std::pow(value - mean, 2);
		}

		double mean_squared_diff = sum_squared_diff / values.size();
		return std::sqrt(mean_squared_diff);
	}

	bool decodeMarker(const cv::Mat& image, Ellipse& ellipse,
		Parameter& param, const float& middle_value, const bool& marker_white)
	{
		unsigned int n_scan_lines = param.code_scan_resolution_per_element * param.code_bits;
		unsigned int max_code_error = ceil(double(param.code_scan_resolution_per_element) / 2.0) - 1;
		double scan_resolutionRad = CV_2PI / double(n_scan_lines);
		std::vector<int> code_pixel(n_scan_lines);
		std::string code_string(param.code_bits, '0');

		// Scanning code ring (TODO: more robust with multiple scan rings)
		double alpha = sin(ellipse.angle - PI_HALF) * param.code_ring_radius;
		double beta = cos(ellipse.angle - PI_HALF) * param.code_ring_radius;
		double x, y;
		cv::Point2f point = cv::Point2f();
		for (unsigned int i = 0; i < n_scan_lines; i++) {
			x = ellipse.a * cos(double(i) * scan_resolutionRad);
			y = ellipse.b * sin(double(i) * scan_resolutionRad);
			point.x = x * beta - y * alpha + ellipse.x;
			point.y = x * alpha + y * beta + ellipse.y;
			auto pixel_value = getSubPixValueWithChecks(image, point);
			if (pixel_value.has_value()) {
				code_pixel[i] = ((pixel_value.value() > middle_value) ^ marker_white) ? 0 : 1;
			}
			else {
				return false; // Code ring outside of image
			}
		}

		// Find first element
		int indexFirstElement = -1;
		for (unsigned int i = 1; i < n_scan_lines && indexFirstElement == -1; i++) {
			if (code_pixel[i - 1] == 0 && code_pixel[i] == 1) {
				indexFirstElement = i;
			}
		}
		if (indexFirstElement == -1) {
			return false;
		}

		// Reduce code bits starting from the first detected element.
		for (int i = 0; i < param.code_bits; i++) {
			unsigned int v = 0;
			for (unsigned int j = 0; j < param.code_scan_resolution_per_element; j++) {
				unsigned int index = indexFirstElement + i * param.code_scan_resolution_per_element + j;
				index = (index < n_scan_lines) ? index : index - n_scan_lines;
				v += code_pixel[index];
			}

			// Check if clear code
			if (v > param.code_scan_resolution_per_element - max_code_error)
				code_string[i] = '1';
			else if (v < max_code_error)
				code_string[i] = '0';
			else
				return false;
		}

		// Decoding number
		int point_id = std::stoi(code_string, nullptr, 2);
		std::string code2 = code_string + code_string;
		for (int i = 1; i < param.code_bits; i++) {
			int _p = std::stoi(code2.substr(i, param.code_bits), nullptr, 2);
			if (point_id > _p)
				point_id = _p;
		}
		ellipse.id = marker_detection::findID(point_id, param.code_bits, param.use_metashape_codes);

		return (ellipse.id > 0);
	}

	void adjustKernelSize(int& kernel)
	{
		kernel = (kernel > 2 && kernel % 2 == 0) ? kernel + 1 : kernel;
	}

	void applyThreshold(const cv::Mat& input, cv::Mat& threshold, const Parameter& param)
	{
		switch (param.threshold_method) {
		case 1: // Otsu threshold
			cv::threshold(input, threshold, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
			break;
		case 2: // Bradley adaptive threshold
			bradley_adaptive_thresholding(input, threshold);
			break;
		case 3: { // Normalize image with Clahe and Otsu threshold
			cv::Mat norm;
			cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
			clahe->setClipLimit(param.clahe_clip_limit);
			clahe->apply(input, norm);
			cv::threshold(norm, threshold, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
			break;
		}
		case 4: { // Normalize image and Otsu threshold
			// copied from: https://stackoverflow.com/questions/14872306/local-normalization-in-opencv
			cv::Mat float_gray, num, den, blur, diff, norm;
			input.convertTo(float_gray, CV_32F, 1.0 / 255.0);

			// numerator = img - gauss_blur(img)
			cv::GaussianBlur(float_gray, blur, cv::Size(0, 0), param.local_norm_sigma_1, 0);
			num = float_gray - blur;

			// denominator = sqrt(gauss_blur(img^2))
			cv::GaussianBlur(num.mul(num), blur, cv::Size(0, 0), param.local_norm_sigma_2, 0);
			cv::pow(blur, 0.5, den);

			// output = numerator / denominator
			diff = num / den;

			// normalize output and threshold
			cv::normalize(diff, norm, 0, 255, cv::NORM_MINMAX, CV_8U);
			cv::threshold(norm, threshold, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
			break;
		}
		default:
			threshold = input;
		}
	}

	void sampleScanlineEllipse(const cv::Mat& image, const float& xCenter, const float& yCenter,
		std::vector<float>& scanline, const float& xPos, const float& yPos, const float& alpha,
		const float& beta, const int& scan_line_steps, const float& min_radius, const float& max_radius)
	{
		scanline.clear();

		// Precompute
		double radius_steps = double(max_radius - min_radius) / double(scan_line_steps);
		double x_beta = xPos * beta;
		double x_alpha = xPos * alpha;
		double y_beta = yPos * beta;
		double y_alpha = yPos * alpha;

		// Sample pixel values
		cv::Point2f point;
		for (int j = 0; j < scan_line_steps; j++) {
			double radius = min_radius + double(j) * radius_steps;
			point.x = xCenter + x_beta * radius - y_alpha * radius;
			point.y = yCenter + x_alpha * radius + y_beta * radius;
			auto v = getSubPixValueWithChecks(image, point);
			if (v.has_value()) {
				scanline.push_back(v.value());
			}
		}
	}

	double estimateModeKDE(const std::vector<double>& data)
	{
		size_t n = data.size();

		// Estimate bandwidth
		double mean = std::accumulate(data.begin(), data.end(), 0.0) / n;
		double std_dev = std::sqrt(std::accumulate(data.begin(), data.end(), 0.0,
			[&](double sum, double x) { return sum + (x - mean) * (x - mean); }) / n);
		double bandwidth = 1.06 * std_dev * std::pow(n, -1.0 / 5.0);  // Silverman's rule

		double max_density = -std::numeric_limits<double>::infinity();
		double mode = 0.0;

		for (const double& x : data) {  // Evaluate density at each data point
			double density = 0.0;

			for (const double& xi : data) {  // Sum up the contributions of all kernels
				double diff = x - xi;
				density += std::exp(-0.5 * (diff * diff) / (bandwidth * bandwidth)) / (std::sqrt(2 * CV_PI) * bandwidth);
			}

			if (density > max_density) {  // Update mode if a higher density is found
				max_density = density;
				mode = x;
			}
		}

		return mode;
	}

} // end namespace marker_detection

