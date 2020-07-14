#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include "cmdline.h"
#include "NdtOctree.h"
#include "visualizer.h"
#include "utils.h"
#include "config.h"
#include <sys/stat.h>
#include <sys/types.h>



#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/common/impl/io.hpp>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/extract_indices.h>


typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloud;

double resolution = 0.05;
double threshold = 0.02;
double delta_d = 0.05;
double delta_thelta = 15.0;

std::string ROOT_DICT;
std::string outPath = "";
std::string fileName = "color_map.json"; // can also use not mandotory parser

nlohmann::json j;

int start_index;

double t1,t2,t3,t4,t5,t6;
config cfg;

void args_parser(int argc, char**argv)
{
    cmdline::parser arg;
    arg.add<std::string>("root", 'r', "root dict", true, "");
    arg.add<std::string>("outpath",'o',"output path",false,"");
    arg.add<int>("start",'s',"start index",false,0);
    arg.parse_check(argc,argv);

    start_index = arg.get<int>("start");
    ROOT_DICT = arg.get<std::string>("root");
    outPath = arg.get<std::string>("outpath");
    if(outPath == "") outPath = ROOT_DICT;
}

PointCloud::Ptr d2cloud_with_semantic(const cv::Mat &depth, 
                                      std::vector<cv::Point> semantic,
                                      Eigen::Matrix3f &intrinsics_matrix)
{
    // Convert rgb and depth image into colorized 3d point cloud
    double const fx_d = intrinsics_matrix(0,0);
    double const fy_d = intrinsics_matrix(1,1);
    double const cx_d = intrinsics_matrix(0,2);
    double const cy_d = intrinsics_matrix(1,2);

    PointCloud::Ptr cloud(new PointCloud);

    for(int i=0; i < semantic.size(); i++){
        int x = semantic[i].x;
        int y = semantic[i].y;
        ushort d = depth.ptr<ushort>(y)[x];
        if (d > 65535*0.75) continue;
        PointT p;
        // calculate xyz coordinate with camera intrinsics paras
        p.z = float (d / 512.);
        p.x = float ((x - cx_d) * p.z / fx_d);
        p.y = float ((y - cy_d) * p.z / fy_d);
        cloud->points.push_back(p);
    }

    cloud->width = semantic.size();
    cloud->height = 1;
    return cloud;
}

double computeloss(PLANE plane)
{
    double sum=0;
    for(int i=0;i<plane.points.size();i++){
        Eigen::Vector3f pt;
        pt << plane.points[i].x,plane.points[i].y,plane.points[i].z;
        float dotproduct = std::fabs((plane.center-pt).dot(plane.normal));
        sum = sum + dotproduct;
    }
    double loss = sum/plane.points.size()*1000;

    return loss;
}

std::vector<std::vector<cv::Point>> readMasks(std::string &semantic_path,nlohmann::json j)
{
    cv::Mat semanticMat = cv::imread(semantic_path);
    std::vector<cv::Scalar> colormap;

    int pos = semantic_path.find_last_of('/');
    std::string semantic_name = semantic_path.substr(pos+1);

    std::vector<std::string> colormap_string;
    colormap_string = j[semantic_name].get<std::vector<std::string>>();
    for(int i=0;i<colormap_string.size();i++){
        std::vector<std::string> color_strings;
        split_string(colormap_string[i],' ', color_strings);
        int b = std::stoi(color_strings[0]);
        int g = std::stoi(color_strings[1]);
        int r = std::stoi(color_strings[2]);
        colormap.push_back(cv::Scalar(b,g,r));
    }

    std::vector<std::vector<cv::Point>> semantic_indices_list(colormap.size());
    for(int i=0;i<colormap.size();i++)
    {
        std::vector<cv::Point> semantic_indices;
        for(int y=0;y<semanticMat.rows;y++){
            for(int x=0;x<semanticMat.cols;x++){
                int b = semanticMat.at<cv::Vec3b>(y,x)[0];
                int g = semanticMat.at<cv::Vec3b>(y,x)[1];
                int r = semanticMat.at<cv::Vec3b>(y,x)[2];
                if(b == colormap[i][0] && g == colormap[i][1] &&  r == colormap[i][2])
                    semantic_indices.push_back(cv::Point(x,y));
            }
        }
        semantic_indices_list[i] = semantic_indices;
    }
    std::sort(semantic_indices_list.begin(), semantic_indices_list.end(), [](const std::vector<cv::Point> & a, const std::vector<cv::Point> & b){ return a.size() > b.size(); });
    return semantic_indices_list;
}


void do_ransac(PointCloud::Ptr cloud, pcl::PointIndices::Ptr inliers, pcl::ModelCoefficients::Ptr coefficients)
{
    // Set Paras with a reasonable theory
    int max_iterations = 500;
    double distance_threshold = 0.1;

    pcl::SACSegmentation<PointT> seg;
    //pcl::SACSegmentationFromNormals<PointT, pcl::Normal> seg;
    seg.setOptimizeCoefficients(true);
    //seg.setModelType (pcl::SACMODEL_NORMAL_PLANE);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(distance_threshold);
    seg.setMaxIterations(max_iterations);
    seg.setInputCloud(cloud);
    seg.segment(*inliers, *coefficients);
}

//getting the inliers/outliers cloud given an pointcloud and indices
PointCloud::Ptr filter_cloud(PointCloud::Ptr cloud, pcl::PointIndices::Ptr indices, bool negative)
{
    PointCloud::Ptr cloud_filtered(new PointCloud);
    pcl::ExtractIndices<PointT> extract;
    extract.setInputCloud(cloud);
    extract.setIndices(indices);
    extract.setNegative(negative);
    extract.filter(*cloud_filtered);
    return cloud_filtered;
}

std::vector<PLANE> ransac_planeseg (const PointCloud::Ptr cloud, const int planeNum)
{
    PointCloud::Ptr cloud_ (new PointCloud);
    std::vector<PointCloud::Ptr> segments;
    std::vector<PLANE> outs;

    pcl::copyPointCloud(*cloud,*cloud_);
    unsigned long int total_points = cloud->size();
    double min_points_threshold = 0.005;
    int i = 0, j=0;

    while(cloud_->points.size()>min_points_threshold*total_points && i < planeNum && j<15)
    {
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        do_ransac(cloud_, inliers, coefficients);
        ///get inliers and outliers
        PointCloud::Ptr outliers_cloud, inliers_cloud;
        inliers_cloud = filter_cloud(cloud_, inliers, false);
        outliers_cloud = filter_cloud(cloud_, inliers, true);

        if(inliers_cloud->points.size()>total_points*0.01)
        {
            PLANE temp;
            for(int i=0; i< inliers_cloud->size(); i++){
                temp.points.push_back(inliers_cloud->points[i]);
            }
            outs.push_back(temp);
            ///prepare for next iter
            cloud_ = outliers_cloud;
            i++;
        }
        j++;
    }

    return outs;
}



std::vector<PLANE> ndtRANSAC_rgbd(std::vector<std::string> &names, config cfg)
{
    std::string pose_path = names[0];
    std::string depth_path= names[2];
    std::string semantic_path= names[3];

    /// Read Pose and Depth,then Generalize Point Cloud
    Eigen::Matrix3f intrinsics_matrix = readIntrinsic(pose_path);
    cv::Mat depthMat = cv::imread(depth_path,cv::IMREAD_ANYDEPTH);

    /// Read Semantic and Generalize Indices
    std::string colormap_path = ROOT_DICT + "/" + fileName;
    std::vector<std::vector<cv::Point>> semantic_maps = readMasks(semantic_path,j);

    std::vector<PLANE> planes;
    for(int i=0;i<semantic_maps.size();i++)
    {
        if(semantic_maps[i].size()< depthMat.cols*depthMat.rows*0.005) {
            continue;
        }
/*
        else if (semantic_maps[i].size() < depthMat.cols*depthMat.rows*0.01){
            PointCloud::Ptr cloud = d2cloud_with_semantic(depthMat,semantic_maps[i],intrinsics_matrix);

            std::vector<PLANE> planes_per_semantic;
            planes_per_semantic = ransac_planeseg (cloud, 3);
            planes.insert(planes.end(),planes_per_semantic.begin(),planes_per_semantic.end());
        }
*/
        else{
            PointCloud::Ptr cloud = d2cloud_with_semantic(depthMat,semantic_maps[i],intrinsics_matrix);

            /// Establish Octree and Rough NDT-Segmentation
            NdtOctree ndtoctree(resolution);
            ndtoctree.setInputCloud(cloud);
            if(ndtoctree.getLeafCount() < 10) continue;
            ndtoctree.computeLeafsNormal();
            ndtoctree.planarSegment(threshold);

            /// NDT-RANSAC
            std::vector<PLANE> planes_per_semantic;
            ndtoctree.ndtRansac(planes_per_semantic,3,delta_d,delta_thelta);
            planes.insert(planes.end(),planes_per_semantic.begin(),planes_per_semantic.end());
        }

    }

    std::sort(planes.begin(), planes.end(),
              [](const PLANE & a, const PLANE & b){ return a.points.size() > b.points.size(); });
    
    return planes;
}

void outputResults(std::vector<PLANE> planes, std::vector<std::string> &names, const std::string &out_path,
                   const int image_id, cv::Size size = cv::Size(550,550))
{ 
    std::string pose_path = names[0];

    /// Read Pose and Depth,then Generalize Point Cloud
    Eigen::Matrix3f intrinsics_matrix = readIntrinsic(pose_path);
    
    std::string out_path_rgb = out_path + "/rgb_out";
    std::string out_path_dep = out_path + "/dep_out";
    std::string out_path_masks = out_path + "/masks";
    std::string output_name_head = "bp" + std::to_string(image_id);

    mkdir(const_cast<char *>(out_path_masks.c_str()), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    mkdir(const_cast<char *>(out_path_dep.c_str()), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    mkdir(const_cast<char *>(out_path_rgb.c_str()), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

    
    /// Visualize Result
    visualizer vs;
    int planeNum = std::min(int(planes.size()),15);
    for(int i=0;i<planeNum;i++){
        cv::Mat mask = vs.projectPlane2Mat(planes[i],intrinsics_matrix);
        cv::resize(mask, mask, size);
        cv::imwrite(out_path_masks + "/" + output_name_head + 
                    "_plane_"+ std::to_string(i) + ".png", mask);
    }

    cv::Mat colorMat = cv::imread(names[1]);
    cv::resize(colorMat, colorMat, size);
    cv::Mat depMat = cv::imread(names[2], cv::IMREAD_ANYDEPTH);
    cv::resize(depMat, depMat, size);

    cv::imwrite(out_path_rgb + "/" + output_name_head + ".jpg", colorMat);
    cv::imwrite(out_path_dep + "/" + output_name_head + "_dep.png", depMat);

}

void outputResults_show(std::vector<PLANE> planes, std::vector<std::string> &names, const std::string &out_path,
                        const int image_id, cv::Size size = cv::Size(550,550))
{
    std::string pose_path = names[0];

    /// Read Pose and Depth,then Generalize Point Cloud
    Eigen::Matrix3f intrinsics_matrix = readIntrinsic(pose_path);

    std::string out_path_rgb = out_path + "/rgb_out";
    std::string out_path_masks = out_path + "/masks";
    std::string out_path_show = out_path + "/show";
    std::string output_name_head = "bp" + std::to_string(image_id);

    mkdir(const_cast<char *>(out_path_masks.c_str()), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    mkdir(const_cast<char *>(out_path_rgb.c_str()), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    mkdir(const_cast<char *>(out_path_show.c_str()), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

    cv::Mat colorMat = cv::imread(names[1]);

    /// Visualize Result
    std::vector<double> losses;
    visualizer vs;
    int planeNum = std::min(int(planes.size()),15);
    std::vector<cv::Mat> masks;
    for(int i=0;i<planeNum;i++){
        cv::Mat mask = vs.projectPlane2Mat(planes[i],intrinsics_matrix);
        cv::imwrite(out_path_masks + "/" + output_name_head +
                    "_plane_"+ std::to_string(i) + ".png", mask);
        losses.push_back(computeloss(planes[i]));
        masks.push_back(mask);
    }

    cv::Mat show = vs.take3in1(masks, colorMat, losses);
    cv::imwrite(out_path_show + "/" + output_name_head+ "_show.jpg", show);
    cv::imwrite(out_path_rgb + "/" + output_name_head + ".jpg", colorMat);
}

int main(int argc, char** argv)
{
    /// Parser Arguments
    args_parser(argc,argv);
    t2 = 0;
    std::cout << outPath << std::endl;

    /// Read Point Cloud
    std::string jsonName = ROOT_DICT + "/" + fileName;
    std::vector<std::vector<std::string>> files = readJSON(jsonName);
    int fileNum = files.size();
    //fileNum = 50;
    std::cout << "Data in Total: " << fileNum << std::endl;

    std::string colormap_path = ROOT_DICT + "/" + fileName;
    std::ifstream file(colormap_path);
    if (file.good()) file >> j;


    /// NDT RANSAC

    int image_id = 100000 + start_index;
    while(image_id-100000 < fileNum)
    {
        t1 = tick();

        std::vector<PLANE> planes = ndtRANSAC_rgbd(files[image_id-100000], cfg);
        outputResults(planes, files[image_id-100000], outPath, image_id);
        t1 = tick()-t1;
        t2 += t1;
        std::cout << "\r" <<  "Processed image[" << image_id - 100000 + 1 << "/ " << fileNum << "], "
                          << "consumed time: " << t1 << " seconds" << std::flush;
        image_id ++;
    }
    std::cout << std::endl  << "time consume per frame: " <<  t2/fileNum << " seconds"  << std::endl;
    std::cout << std::endl  << "total time consume: " <<  t2/3600 << " hours"  << std::endl;

    return (0);
}
