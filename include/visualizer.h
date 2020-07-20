#ifndef VISUALIZER_H
#define VISUALIZER_H

#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <nlohmann/json.hpp>
#include "NdtOctree.h"

class visualizer {

    typedef pcl::PointXYZ PointT;
    typedef pcl::PointCloud<PointT> PointCloud;

public:

    visualizer();

    ~visualizer();

    cv::Mat projectPlane2Mat(PLANE &plane, Eigen::Matrix3f camera_intrinsic);

    cv::Mat take3in1(std::vector<cv::Mat> masks, cv::Mat raw, std::vector<double> losses);

    cv::Mat take3in1_tum(std::vector<cv::Mat> masks, cv::Mat raw);

    cv::Mat maskSuperposition(std::vector<cv::Mat> masks, bool color_or_gray);

    cv::Mat applyMask (cv::Mat raw, cv::Mat mask, double transparency);
   
    cv::Mat projectPointCloud2Mat(const PointCloud::Ptr cloud, Eigen::Matrix3f camera_intrinsic);

private:
    static std::vector<cv::Scalar> colors_list;

    int round_double(double a);
};


#endif //VISUALIZER_H



