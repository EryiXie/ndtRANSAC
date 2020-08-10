#include "visualizer.h"
#include "SettingReader.h"

std::vector<cv::Scalar> visualizer::ColorPalette;
cv::Size  visualizer::single_frameSize;

visualizer::visualizer(cv::Size frameSize) 
{
    ColorPalette.resize(20);
    ColorPalette[0] = cv::Scalar(250, 211, 157);
    ColorPalette[1] = cv::Scalar(181, 111, 31);
    ColorPalette[2] = cv::Scalar(146, 214, 252);
    ColorPalette[3] = cv::Scalar(235, 255, 250);
    ColorPalette[4] = cv::Scalar(92, 36, 22);
    ColorPalette[5] = cv::Scalar(108, 163, 86);
    ColorPalette[6] = cv::Scalar(79, 136, 126);
    ColorPalette[7] = cv::Scalar(79, 195, 119);
    ColorPalette[8] = cv::Scalar(163, 184, 170);
    ColorPalette[9] = cv::Scalar(200, 237, 255);
    ColorPalette[10] = cv::Scalar(60, 2, 112);
    ColorPalette[11] = cv::Scalar(232, 206, 239);
    ColorPalette[12] = cv::Scalar(202, 249, 218);
    ColorPalette[13] = cv::Scalar(87, 233, 255);
    ColorPalette[14] = cv::Scalar(93, 117, 242);
    ColorPalette[15] = cv::Scalar(64, 64, 64);
    ColorPalette[16] = cv::Scalar(0, 102, 51);
    ColorPalette[17] = cv::Scalar(0, 64, 212);
    ColorPalette[18] = cv::Scalar(90, 128, 64);
    ColorPalette[19] = cv::Scalar(128, 128, 32);

    single_frameSize = frameSize;
}

visualizer::~visualizer() {
    
}

cv::Mat visualizer::projectPlane2Mat(const PLANE &plane, const Eigen::Matrix3f &camera_intrinsic)
{
    cv::Mat mask = cv::Mat::zeros(single_frameSize, CV_8UC1);
    for (unsigned int i = 0; i < plane.points.size(); i++) {
        double x = plane.points[i].x;
        double y = plane.points[i].y;
        double z = plane.points[i].z;
        int u = round_double(x * camera_intrinsic(0, 0) / z + camera_intrinsic(0, 2));
        int v = round_double(y * camera_intrinsic(1, 1) / z + camera_intrinsic(1, 2));
        mask.at<uchar>(v,u) = 255;
    }

    cv::Mat element = getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
    cv::erode(mask, mask, element);
    cv::dilate(mask, mask, element); 
    cv::dilate(mask, mask, element);
    cv::erode(mask, mask, element);
   
    return mask;
}

cv::Mat visualizer::maskSuperposition(const std::vector<cv::Mat> &masks, const bool &color_or_gray)
{
    if (color_or_gray)
    {
        cv::Mat allmask = cv::Mat::zeros(masks[0].rows, masks[0].cols, CV_8UC3);
        for (unsigned int index = 0; index < masks.size(); index++) {
            for (int y = 0; y < allmask.rows; y++) {
                for (int x = 0; x < allmask.cols; x++) {
                    if (masks[index].at<uchar>(y,x)==255) {
                        allmask.at<cv::Vec3b>(y, x)[0] = ColorPalette[index][0];
                        allmask.at<cv::Vec3b>(y, x)[1] = ColorPalette[index][1];
                        allmask.at<cv::Vec3b>(y, x)[2] = ColorPalette[index][2];
                    }
                }
            }
        }
        return allmask;
    }
    else {
        cv::Mat allmask = cv::Mat::zeros(masks[0].rows, masks[0].cols, CV_8UC1);
        for (unsigned int index = 0; index < masks.size(); index++) {
            for (int y = 0; y < allmask.rows; y++) {
                for (int x = 0; x < allmask.cols; x++) {
                    if (masks[index].at<uchar>(y,x)==255) {
                        allmask.at<uchar>(y, x) = index*8 + 32;
                        }
                }
            }
        }
        return allmask;
    }
}

cv::Mat visualizer::applyMask (const cv::Mat &raw, const cv::Mat &mask, const double &transparency)
{
    cv::Mat masked = cv::Mat::zeros(raw.size(),CV_8UC3);

    for (int y = 0; y < masked.rows; y++)
    {
        for (int x = 0; x < masked.cols; x++)
        {
            cv::Vec3b color = mask.at<cv::Vec3b>(cv::Point(x, y));
            if (color[0] != 0 && color[1] != 0 && color[2] != 0)
            {
                masked.at<cv::Vec3b>(y,x)[0] = raw.at<cv::Vec3b>(cv::Point(x, y))[0]*(1-transparency) + raw.at<cv::Vec3b>(cv::Point(x, y))[0]*transparency;
                masked.at<cv::Vec3b>(y,x)[1] = raw.at<cv::Vec3b>(cv::Point(x, y))[1]*(1-transparency) + raw.at<cv::Vec3b>(cv::Point(x, y))[1]*transparency;
                masked.at<cv::Vec3b>(y,x)[2] = raw.at<cv::Vec3b>(cv::Point(x, y))[2]*(1-transparency) + raw.at<cv::Vec3b>(cv::Point(x, y))[2]*transparency;
            }
        }
        
    }
    return raw*(1-transparency) + mask*transparency;
}

int visualizer::round_double(const double &a) 
{ 
    return (a > 0.0) ? (a + 0.5) : (a - 0.5); 
}

cv::Mat visualizer::projectPointCloud2Mat(const PointCloud::Ptr &cloud, const Eigen::Matrix3f &camera_intrinsic)
{
    cv::Mat mask = cv::Mat::zeros(single_frameSize, CV_8UC1);
    for (unsigned int i = 0; i < cloud->points.size(); i++) {
        double x = cloud->points[i].x;
        double y = cloud->points[i].y;
        double z = cloud->points[i].z;
        int u = round_double(x * camera_intrinsic(0, 0) / z + camera_intrinsic(0, 2));
        int v = round_double(y * camera_intrinsic(1, 1) / z + camera_intrinsic(1, 2));
        mask.at<uchar>(v,u) = 255;
    }

    cv::Mat element = getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::dilate(mask, mask, element);
    cv::erode(mask, mask, element);
    return mask;
}



cv::Mat visualizer::draw_colormap_blend_labels(const std::vector<cv::Mat> &masks, const cv::Mat &raw)
{
    int masksNum = masks.size();
    cv::Mat mask = maskSuperposition(masks,true);
    cv::Mat masked = applyMask (raw, mask, 0.2);

    cv::Mat all = cv::Mat::zeros(mask.rows, mask.cols*2 + 300, CV_8UC3);
    cv::Rect mask_rect = cv::Rect(0,0, mask.cols, mask.rows);
    mask.copyTo(all(mask_rect));
    cv::Rect masked_rect = cv::Rect(mask.cols,0, mask.cols, mask.rows);
    masked.copyTo(all(masked_rect));

    for (int index = 0; index < masksNum; index++) {
        cv::Point pt = cv::Point(mask.cols*2 + 30, 30 + index * 27);
        cv::rectangle(all, cv::Point(pt.x - 17, pt.y - 17), cv::Point(pt.x + 250, pt.y + 17),
                        cv::Scalar(255, 255, 255), -1);
        std::string text = std::to_string(index + 1);
        cv::putText(all, text, cv::Point(pt.x + 25, pt.y + 5), cv::FONT_HERSHEY_SIMPLEX,
                    0.6, cv::Scalar(0,0,0), 2);
        cv::circle(all, pt, 10, cv::Scalar(0,0,0), -1);
        cv::circle(all, pt, 9, ColorPalette[index], -1);
    }

    return all;
}