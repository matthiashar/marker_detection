#ifndef MARKERDETECTION_H
#define MARKERDETECTION_H

#include <optional>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace marker_detection {
	// = PI * 0.5
	static const double PI_HALF = CV_PI * 0.5;

	struct EllipseDebug
	{
		/// Initial connected edge pixel
		std::vector<cv::Point> edge_points_initial;

		/// Sub pixel edge points (optional)
		std::vector<cv::Point2f> edge_points_sub_pixel;

		/// Edge points after robust ellipse fit (optional)
		std::vector<cv::Point2f> edge_points_robust;

		/// Mean value and rmse of marker
		double marker_value, marker_rmse;			

		/// Mean value and rmse of area around the marker
		double surrounding_value, surrounding_rmse;
	};
	struct DetectionDebug
	{
		/// Threshold image
		cv::Mat threshold;

		/// Edge image
		cv::Mat edge;

		/// Connected edge pixel
		std::vector<std::vector<cv::Point>> connected_edges;
	};

	/**
	 * @brief struct for ellipse
	 * x, y = center
	 * a, b = semi-major and semi-minor axis (a > b)
	 * angle = angle in rad
	 * id = point id, for unknown id == -1
	 * madn = Normalized Median Absolute Deviation (will only be computed when robust_ellipse_fit = true)
	 */
	struct Ellipse
	{
		Ellipse() : x(0.0), y(0.0), a(0.0), b(0.0), angle(0.0), id(-1), mdan(-1) {}
		Ellipse(int id, double x, double y) :
        id(id), x(x), y(y), a(0.0), b(0.0), angle(0.0), mdan(-1) {}
		Ellipse(double x, double y, double a, double b, double angle) :
        x(x), y(y), a(a), b(b), angle(angle), id(-1), mdan(-1) {}
		double x, y, a, b, angle, mdan;
		int id;
		std::shared_ptr<EllipseDebug> debug;
		cv::Point2d point() const {
			return cv::Point2d(x, y);
		}
		bool isValid() const {
			return (a > 0 && b > 0 && a >= b);
		}
	};

	/**
	 * @brief The Parameter struct
	 *
	 */
	struct Parameter
	{
		Parameter() :
			ellipse_fit_type(0),
			marker_min_diameter(8),
			marker_max_diameter(300),
			code_scan_resolution_per_element(5),
			code_ring_radius(2.5),
			code_bits(14),
			use_metashape_codes(false),
			marker_min_contrast(30),
			return_uncoded_marker(true),
			detect_coded_marker(true),
			robust_ellipse_fit(true),
			median_blur_kernel(5),
			blur_kernel(-1),
			max_ellipse_ratio(3.0),
			canny_threshold1(40),
			canny_threshold2(100),
			threshold_method(0),
			clahe_clip_limit(5),
			local_norm_sigma_1(5),
			local_norm_sigma_2(25),
			sub_pixel_method(1),
			sub_pixel_scan_lines(50),
			sub_pixel_iterations(2),
			min_distance_closest_point(1.0),
			edge_method(0),
			adaptive_threshold_method(cv::ADAPTIVE_THRESH_GAUSSIAN_C),
			adaptive_threshold_C(3),
			adaptive_threshold_block_size(7),
			marker_contrast_consistency(0.60),
			max_marker_value_rmse(15),
			max_surrounding_value_rmse(100)
    {}

		/// Type of fitting ellipse: 0: cv::fitEllipse(), 1: cv::fitEllipseAMS(), 2: cv::fitEllipseDirect()
		int ellipse_fit_type;
		/// Minimum diameter of marker in pixel
		unsigned int marker_min_diameter;
		/// Maximum diameter of marker in pixel
		unsigned int marker_max_diameter;
		/// Number of measurements per code element (default: 5)
		unsigned int code_scan_resolution_per_element;
		/// Radius of code ring around marker, where marker radius=1 (default: 2.5)
		double code_ring_radius;
		/// Type of code, must be even number (default: 14)
		int code_bits;
		/// Use Metashape codes, only for 12 and 14 bit (default: false)
		bool use_metashape_codes;
		/// Minimum contrast between marker and surrounding area
		unsigned int marker_min_contrast;
		/// Set true if uncoded markers should be returned
		bool return_uncoded_marker;
		/// Set true if coded markers should be searched for
		bool detect_coded_marker;
		/// If true outlier edge points will be removed during ellipse fit
		bool robust_ellipse_fit;
		/// Kernel for median blur before initial marker detection
		int median_blur_kernel;
		/// Kernel for blur before initial marker detection
		int blur_kernel;
		/// Maximum ratio between minor and major axis of ellipse (a/b)
		double max_ellipse_ratio;
		/// Canny threshold1 for edge detection
		unsigned int canny_threshold1;
		/// Canny threshold2 for edge detection
		unsigned int canny_threshold2;
		/// Threshold method (0 - None, 1 - Otsu, 2 - Bradley, 3 - Clahe, 4 - Xiong)
		int threshold_method;
		/// Clahe clip
		int clahe_clip_limit;
		/// Sigma 1 for Local Normalization
		int local_norm_sigma_1;
		/// Sigma 2 for Local Normalization
		int local_norm_sigma_2;
		/// Sub-pixel method: 0 - None, 1 - Star operator, 2 = Zhou
		int sub_pixel_method;
		/// Number of sub pixel scan lines
		unsigned int sub_pixel_scan_lines;
		/// Number of iterations for adjusting sub pixel position
		int sub_pixel_iterations;
		/// If true, uncoded points will also be returned if they are close to coded points.
		double min_distance_closest_point;
		/// Method for extracting edge. (0 = Canny, 1 = cv::adaptiveThreshold())
		int edge_method;
		/// Parameter for cv::adaptiveThreshold(), see OpenCV documentation.
		int adaptive_threshold_method;
		/// Parameter for cv::adaptiveThreshold(), see OpenCV documentation.
		int adaptive_threshold_C;
		/// Parameter for cv::adaptiveThreshold(), see OpenCV documentation.
		int adaptive_threshold_block_size;
		/// Consistency of contrast at the edge of the marker. 0 = inconsistent contrast around marker, 1 = same contrast at edge of marker
		double marker_contrast_consistency;
		/// Maximum rmse of the marker pixel values
		double max_marker_value_rmse;
		/// Maximum rmse of the pixel values surrounding the marker
		double max_surrounding_value_rmse;
	};

	/**
	 * @brief  Fast method for detecting and decoding markers
	 * @param image - Input: image (CV_8U)
	 * @param markers - Output: detected markers
	 * @param param - Detection Parameter
	 * @param debug - Optional output of additional information
	 */
	void detectAndDecode(const cv::Mat& image, std::vector<Ellipse>& markers,
		Parameter param = Parameter(), DetectionDebug* debug = nullptr);

	/**
	 * @brief Moment preservation (Luhmann 2018, S 453ff.) Note: The Center of the first Element in the Vector is 0.
	 */
	template <typename T>
	void momentPreservation(const std::vector<T>& values, float& pos, float& h1, float& h2);

	/**
	 * @brief Bilinear interpolation of pixel value. Note: No checks are performed!
	 * @param image - Input single-channel image (CV_8U).
	 * @param point - Subpixel point
	 * @return Interpolated intensity value.
	 */
	float getSubPixValue(const cv::Mat& image, const cv::Point2f& point);

	/**
	 * @brief Bilinear interpolation of pixel value. Note: Same as getSubPixValue, but with checks.
	 * @param image - Input single-channel image (CV_8U).
	 * @param point - Subpixel point
	 * @return Interpolated intensity value or std::nullopt if out of image bounds.
	 */
	std::optional<float> getSubPixValueWithChecks(const cv::Mat& image, const cv::Point2f& point);

	/**
	 * @brief Fitting points to ellipse using opencv methods.
	 * @param edge_points - e.g. std::vector<cv::Point2f>
	 * @param type - 0=cv::fitEllipse(), 1=cv::fitEllipseAMS(), 2=cv::fitEllipseDirect()
	 * @return adjusted ellipse
	 */
	void fitEllipse(cv::InputArray& edge_points, Ellipse& ellipse, int type);

	/**
	 * @brief Robust check if points are on ellipse (see: https://doi.org/10.1016/j.isprsjprs.2021.04.010)
	 * @param initial_ellipse - initial ellipse, will be updated
	 * @param edge_points - e.g. std::vector<cv::Point2f>
	 * @param type - 0=cv::fitEllipse(), 1=cv::fitEllipseAMS(), 2=cv::fitEllipseDirect()
	 * @return true if robust adjustment was successful
	 */
	bool fitEllipseRobust(Ellipse& ellipse, cv::InputArray& edge_points,
		int type, int max_iterations = 10, double threshold = 2.45);

	/**
	 * @brief subPixelMeasurement
	 * @param image
	 * @param in
	 * @param out
	 * @param param
	 * @return
	 */
	bool subPixelMeasurement(const cv::Mat& image, Ellipse& ellipse,
		Parameter& param, double marker_value, double marker_around);

	/**
	 * @brief Subpixel marker detection using the star operator.
	 * @param image
	 * @param ellipse
	 * @param sub_pixel_scan_lines
	 * @param min_contrast
	 * @return
	 */
	bool starOperator(const cv::Mat& image, Ellipse& ellipse, Parameter& param,
		const double& marker_value, const double& marker_around);

	/**
	 * @brief zhouOperator
	 * @param image
	 * @param in
	 * @param out
	 * @return
	 */
	bool zhouOperator(const cv::Mat& image, Ellipse& ellipse,
		double marker_value, double marker_around,
		const Parameter& param = marker_detection::Parameter());

	/**
	 * @brief Detect and return connected edge points
	 * @param image
	 * @param markers
	 * @param param
	 * @param debug
	 */
	void findConnectedEdgePoints(const cv::Mat& image, std::vector<std::vector<cv::Point>>& markers,
		Parameter& param, DetectionDebug* debug = nullptr);

	/**
	 * @brief Search marker at specific location in image and detect with sub-pixel accuracy
	 * @param image
	 * @param center_guess
	 * @param param
	 * @return true if marker was found
	 */
	bool searchMarker(const cv::Mat& image, Ellipse& ell, Parameter& param);

	/**
	 * @brief Adaptive Thresholding Methode (D. Bradley and G. Roth, “Adaptive Thresholding using the Integral Image,” Journal of Graphics Tools, vol. 12, Art. no. 2, Jan. 2007, doi: 10.1080/2151237x.2007.10129236).
	 * @param in
	 * @param out
	 */
	void bradley_adaptive_thresholding(const cv::Mat& in, cv::Mat& out);

	/**
	 * @brief Method for calculating the average value of the marker and the surrounding area
	 * @param e
	 * @param image
	 * @param marker_value
	 * @param marker_rmse
	 * @param surrounding_value
	 * @param surrounding_rmse
	*/
	void checkMarkerSurrounding(const Ellipse& e, const cv::Mat& image,
		double& marker_value, double& marker_rmse,
		double& surrounding_value, double& surrounding_rmse);

	/**
	 * @brief Applies a thinning iteration to a binary image.
	 * copy from: https://github.com/opencv/opencv_contrib/blob/4.x/modules/ximgproc/src/thinning.cpp
	 */
	static void thinningIteration(cv::Mat img, int iter, int thinningType);

	/**
	 * @brief Function for calculation the median.
	 * @param values - Input vector of double values.
	 * @return Median of the input vector.
	 */
	double calculateMedian(std::vector<double> values);

	/**
	 * @brief Function for calculation the rmse.
	 * @param values - Input vector of double values.
	 * @return RMSE of the input vector.
	 */
	double calculateRMSE(std::vector<double>& values);

	/**
	 * @brief Function to detect the code ring around an ellipse and decode the ID.
	 * @param image - Image
	 * @param ellipse - Ellipse
	 * @param param - Parameter
	 * @param middle_value - Average pixel value of marker
	 * @param marker_white - True if white marker on black background.
	 * @return Function will return true if ID was found.
	 */
	bool decodeMarker(const cv::Mat& image, Ellipse& ellipse,
		Parameter& param, const float& middle_value, const bool& marker_white);

	/**
	 * @brief Update kernel size to odd number (if larger than 2).
	 * @param kernel
	 */
	void adjustKernelSize(int& kernel);

	/**
	 * @brief Threshold image
	 * @param input - Input image
	 * @param threshold - Output theshold image
	 * @param param
	 */
	void applyThreshold(const cv::Mat& input, cv::Mat& threshold, const Parameter& param);

	/**
	 * @brief Sample pixel values from a image on a line from the center of an ellipse.
	 * @param image - Input image
	 * @param xCenter - Center of ellipse in x
	 * @param yCenter - Center of ellipse in y
	 * @param scanline - Output pixel values
	 * @param xPos - Precomputed value (ellipse.a * cos(scan_angle))
	 * @param yPos - Precomputed value (ellipse.b * sin(scan_angle))
	 * @param alpha - Precomputed value (sin(ellipse.angle - PI_HALF))
	 * @param beta - Precomputed value (cos(ellipse.angle - PI_HALF))
	 * @param scan_line_steps - Number of steps on scanline
	 * @param min_radius - Start radius for sampling
	 * @param max_radius - End radius for sampling
	 */
	void sampleScanlineEllipse(const cv::Mat& image, const float& xCenter, const float& yCenter,
		std::vector<float>& scanline, const float& xPos, const float& yPos, const float& alpha,
		const float& beta, const int& scan_line_steps, const float& min_radius, const float& max_radius);

	/**
	 * @brief Function to estimate the mode of a dataset using Kernel Density Estimation (KDE).
	 * @param data 
	 * @return mod
	 */
	double estimateModeKDE(const std::vector<double>& data);

} // end namespace marker_detection

#endif // MARKERDETECTION_H
