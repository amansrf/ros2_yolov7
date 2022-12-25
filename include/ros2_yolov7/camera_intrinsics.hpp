#include <opencv2/opencv.hpp>

// Struct to hold the intrinsic parameters of a camera
struct CameraIntrinsics {
    cv::Mat cameraMatrix;
    cv::Mat distortionCoefficients;
};

// Define the intrinsic parameters for 6 cameras
static const CameraIntrinsics vimba_front_left_center_intrinsics = {
    (cv::Mat_<double>(3,3) << 1732.571708, 0.000000, 549.797164, 0.000000, 1731.274561, 295.484988, 0.000000, 0.000000, 1.000000),
    (cv::Mat_<double>(1,5) << -0.272455, 0.268395, -0.005054, 0.000391, 0.000000)
};

static const CameraIntrinsics vimba_front_right_center_intrinsics = {
    (cv::Mat_<double>(3,3) << 1687.074326, 0.000000, 553.239937, 0.000000, 1684.357776, 405.545108, 0.000000, 0.000000, 1.000000),
    (cv::Mat_<double>(1,5) << -0.245135, 0.211710, -0.001225, 0.001920, 0.000000)
};

static const CameraIntrinsics vimba_front_left_intrinsics = {
    (cv::Mat_<double>(3,3) << 251.935177, 0.000000, 260.887279, 0.000000, 252.003440, 196.606218, 0.000000, 0.000000, 1.000000),
    (cv::Mat_<double>(1,5) << -0.181920, 0.050231, -0.000085, 0.001209, 0.000000)
};

static const CameraIntrinsics vimba_front_right_intrinsics = {
    (cv::Mat_<double>(3,3) << 250.433937, 0.000000, 257.769878, 0.000000, 250.638855, 186.572321, 0.000000, 0.000000, 1.000000),
    (cv::Mat_<double>(1,5) << -0.185526, 0.050462, -0.000342, 0.000089, 0.000000)
};

static const CameraIntrinsics vimba_rear_left_intrinsics = {
    (cv::Mat_<double>(3,3) << 250.596029, 0.000000, 258.719088, 0.000000, 250.571031, 190.881036, 0.000000, 0.000000, 1.000000),
    (cv::Mat_<double>(1,5) << -0.179032, 0.047829, 0.000395, -0.000710, 0.000000)
};

static const CameraIntrinsics vimba_rear_right_intrinsics = {
    (cv::Mat_<double>(3,3) << 251.353372, 0.000000, 257.997245, 0.000000, 251.239242, 194.330435, 0.000000, 0.000000, 1.000000),
    (cv::Mat_<double>(1,5) << -0.192458, 0.059495, -0.000126, 0.000092, 0.000000)
};

