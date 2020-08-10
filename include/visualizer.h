#ifndef VISUALIZER_H
#define VISUALIZER_H

#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <nlohmann/json.hpp>
#include "NdtOctree.h"

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloud;

class visualizer {

public:

    visualizer(cv::Size frameSize);

    ~visualizer();

    cv::Mat projectPlane2Mat(const PLANE &plane, const Eigen::Matrix3f &camera_intrinsic);

    cv::Mat draw_colormap_blend_labels(const std::vector<cv::Mat> &masks, const cv::Mat &raw);

    cv::Mat maskSuperposition(const std::vector<cv::Mat> &masks, const bool &color_or_gray);

    cv::Mat applyMask (const cv::Mat &raw, const cv::Mat &mask, const double &transparency);
   
    cv::Mat projectPointCloud2Mat(const PointCloud::Ptr &cloud, const Eigen::Matrix3f &camera_intrinsic);

private:
    static std::vector<cv::Scalar> ColorPalette;

    static cv::Size single_frameSize;

    int round_double(const double &a);
};


#endif //VISUALIZER_H



