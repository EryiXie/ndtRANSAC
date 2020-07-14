#ifndef VISUALIZER_H
#define VISUALIZER_H

#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <nlohmann/json.hpp>




class visualizer {

    typedef pcl::PointXYZ PointT;
    typedef pcl::PointCloud<PointT> PointCloud;

public:

    visualizer() {
        colors_list.resize(20);
        colors_list[0] = cv::Scalar(250, 211, 157);
        colors_list[1] = cv::Scalar(181, 111, 31);
        colors_list[2] = cv::Scalar(146, 214, 252);
        colors_list[3] = cv::Scalar(235, 255, 250);
        colors_list[4] = cv::Scalar(92, 36, 22);
        colors_list[5] = cv::Scalar(108, 163, 86);
        colors_list[6] = cv::Scalar(79, 136, 126);
        colors_list[7] = cv::Scalar(79, 195, 119);
        colors_list[8] = cv::Scalar(163, 184, 170);
        colors_list[9] = cv::Scalar(244, 237, 235);
        colors_list[10] = cv::Scalar(60, 2, 112);
        colors_list[11] = cv::Scalar(232, 206, 239);
        colors_list[12] = cv::Scalar(202, 249, 218);
        colors_list[13] = cv::Scalar(87, 233, 255);
        colors_list[14] = cv::Scalar(93, 117, 242);
        colors_list[15] = cv::Scalar(64, 64, 64);
        colors_list[16] = cv::Scalar(0, 102, 51);
        colors_list[17] = cv::Scalar(0, 64, 212);
        colors_list[18] = cv::Scalar(90, 128, 64);
        colors_list[19] = cv::Scalar(128, 128, 32);
    }

    ~visualizer() {}

    cv::Mat projectPlane2Mat(PLANE &plane, Eigen::Matrix3f camera_intrinsic)
    {
        //cv::Mat mask = cv::Mat::zeros(int(camera_intrinsic(0, 2) * 2), int(camera_intrinsic(1, 2) * 2), CV_8UC1);
        cv::Mat mask = cv::Mat::zeros(480, 640, CV_8UC1);
        for (unsigned int i = 0; i < plane.points.size(); i++) {
            double x = plane.points[i].x;
            double y = plane.points[i].y;
            double z = plane.points[i].z;
            int u = round_double(x * camera_intrinsic(0, 0) / z + camera_intrinsic(0, 2));
            int v = round_double(y * camera_intrinsic(1, 1) / z + camera_intrinsic(1, 2));
            mask.at<uchar>(v,u) = 255;
        }

        cv::Mat element = getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::dilate(mask, mask, element);
        cv::erode(mask, mask, element);
        cv::erode(mask, mask, element);
        cv::dilate(mask, mask, element);
        return mask;
    }

    cv::Mat take3in1(std::vector<cv::Mat> masks, cv::Mat raw, std::vector<double> losses)
    {
        int masksNum = masks.size();
        cv::Mat mask = maskSuperposition(masks,true);
        cv::Mat masked = applyMask (raw, mask, 0.3);

        cv::Mat all = cv::Mat::zeros(mask.rows, mask.cols*2 + 300, CV_8UC3);
        cv::Rect mask_rect = cv::Rect(0,0, mask.cols, mask.rows);
        mask.copyTo(all(mask_rect));
        cv::Rect masked_rect = cv::Rect(mask.cols,0, mask.cols, mask.rows);
        masked.copyTo(all(masked_rect));

        for (int index = 0; index < masksNum; index++) {
            cv::Point pt = cv::Point(mask.cols*2 + 30, 30 + index * 35);
            cv::rectangle(all, cv::Point(pt.x - 17, pt.y - 17), cv::Point(pt.x + 250, pt.y + 17),
                          cv::Scalar(255, 255, 255), -1);
            std::string text = std::to_string(index + 1) + "  " + std::to_string(losses[index]) ;
            cv::putText(all, text, cv::Point(pt.x + 25, pt.y + 12), cv::FONT_HERSHEY_SIMPLEX,
                        1, cv::Scalar(0,0,0), 2);
            cv::circle(all, pt, 16, cv::Scalar(0,0,0), -1);
            cv::circle(all, pt, 15, colors_list[index], -1);
        }

        return all;
    }

    cv::Mat take3in1_tum(std::vector<cv::Mat> masks, cv::Mat raw)
    {
        //unsigned int masksNum = masks.size();
        cv::Mat mask = maskSuperposition(masks,true);
        cv::Mat masked = applyMask (raw, mask, 0.15);

        cv::Mat all = cv::Mat::zeros(mask.rows, mask.cols*2, CV_8UC3);
        cv::Rect mask_rect = cv::Rect(0,0, mask.cols, mask.rows);
        mask.copyTo(all(mask_rect));
        cv::Rect masked_rect = cv::Rect(mask.cols,0, mask.cols, mask.rows);
        masked.copyTo(all(masked_rect));
        return all;
    }


    cv::Mat maskSuperposition(std::vector<cv::Mat> masks, bool color_or_gray)
    {
        if (color_or_gray)
        {
            cv::Mat allmask = cv::Mat::zeros(masks[0].rows, masks[0].cols, CV_8UC3);
            for (unsigned int index = 0; index < masks.size(); index++) {
               for (int y = 0; y < allmask.rows; y++) {
                  for (int x = 0; x < allmask.cols; x++) {
                      if (masks[index].at<uchar>(y,x)==255) {
                          allmask.at<cv::Vec3b>(y, x)[0] = colors_list[index][0];
                          allmask.at<cv::Vec3b>(y, x)[1] = colors_list[index][1];
                          allmask.at<cv::Vec3b>(y, x)[2] = colors_list[index][2];
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

    cv::Mat applyMask (cv::Mat raw, cv::Mat mask, double transparency)
    {
        cv::Mat masked = cv::Mat::zeros(raw.rows,raw.cols,CV_8UC3);

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
   

    cv::Mat projectPointCloud2Mat(const PointCloud::Ptr cloud, std::vector<int> indices, Eigen::Matrix3f camera_intrinsic)
    {
        cv::Mat mask = cv::Mat::zeros(int(camera_intrinsic(0, 2) * 2), int(camera_intrinsic(1, 2) * 2), CV_8UC3);
        for (unsigned int i = 0; i < indices.size(); i++) {
            int index = indices[i];
            double x = cloud->points[index].x;
            double y = cloud->points[index].y;
            double z = cloud->points[index].z;
            int u = round_double(x * camera_intrinsic(0, 0) / z + camera_intrinsic(0, 2));
            int v = round_double(y * camera_intrinsic(1, 1) / z + camera_intrinsic(1, 2));
            mask.at<cv::Vec3b>(v, u)[0] = 255;
            mask.at<cv::Vec3b>(v, u)[1] = 255;
            mask.at<cv::Vec3b>(v, u)[2] = 255;
        }

        cv::Mat element = getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::dilate(mask, mask, element);
        cv::erode(mask, mask, element);

        return mask;
    }

private:
    std::vector<cv::Scalar> colors_list;
    int round_double(double a) { return (a > 0.0) ? (a + 0.5) : (a - 0.5); }

};


#endif //VISUALIZER_H



