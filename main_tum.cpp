#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include "cmdline.h"
#include "NdtOctree.h"
#include "visualizer.h"
#include "utils.h"
#include "SettingReader.h"
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

std::string ROOT_DICT;
std::string outPath = "";
std::string fileName = "tumval.json";

int start_index;

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

std::vector<PLANE> ndtRANSAC(const PointCloud::Ptr &cloud, config cfg, int max_plane_per_cloud)
{
    std::vector<PLANE> planes;
    /// Establish Octree and Rough NDT-Segmentation
    NdtOctree ndtoctree(cfg.resolution);
    ndtoctree.setInputCloud(cloud);
    ndtoctree.computeLeafsNormal();
    ndtoctree.planarSegment(cfg.threshold);
    /// NDT-RANSAC
    ndtoctree.ndtRansac(planes, max_plane_per_cloud, cfg.delta_d, cfg.delta_thelta);
    std::sort(planes.begin(), planes.end(),
              [](const PLANE & a, const PLANE & b){ return a.points.size() > b.points.size(); });
    return planes;
}

void outputResults(config cfg, TUMReader dataset, std::vector<PLANE> planes, int index)
{
    if(planes.size()>0) {
        /// Visualize Result
        visualizer vs;
        std::vector<cv::Mat> masks;
        std::string out_path_rgb = outPath + "/rgb_out";
        std::string out_path_dep = outPath + "/dep_out";
        std::string out_path_masks = outPath + "/masks";
        std::string out_path_show = outPath + "/show";
        std::string colorPath = ROOT_DICT + "/" + dataset.rgbList[index];
        std::string depthPath = ROOT_DICT + "/" + dataset.depthList[index];

        int planeNum = std::min(int(planes.size()),cfg.max_output_planes);

        for (int i = 0; i < planeNum; i++) {
            masks.push_back(vs.projectPlane2Mat(planes[i], dataset.intrinsic));
        }

        if(cfg.use_indiv_masks){
            mkdir(const_cast<char *>(out_path_masks.c_str()), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
            for (int i = 0; i < planeNum; i++)
                cv::imwrite(out_path_masks + "/" + cfg.output_head_name + "_"  + std::to_string(10000+index)
                        + "_mask_" + std::to_string(i) +".png", masks[i]);
        }

        if(cfg.use_present_sample)
        {
            mkdir(const_cast<char *>(out_path_show.c_str()), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
            cv::Mat colorMat = cv::imread(colorPath);
            cv::Mat show =  vs.take3in1_tum(masks, colorMat);
            cv::imwrite(out_path_show + "/" + cfg.output_head_name + "_" + std::to_string(index) + "_show.jpg", show);
        }
        if(cfg.use_output_resize)
        {
            mkdir(const_cast<char *>(out_path_dep.c_str()), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
            mkdir(const_cast<char *>(out_path_rgb.c_str()), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
            cv::Mat colorMat = cv::imread(colorPath);
            cv::resize(colorMat, colorMat, cv::Size(550,550));
            cv::Mat depMat = cv::imread(depthPath, cv::IMREAD_ANYDEPTH);
            cv::resize(depMat, depMat, cv::Size(550,550));
            cv::imwrite(out_path_rgb + "/" + cfg.output_head_name + "_" + std::to_string(index) + ".jpg", colorMat);
            cv::imwrite(out_path_dep + "/" + cfg.output_head_name + "_" + std::to_string(index) + "_dep.png", depMat);
        }

    }
    else{
        std::string out_path_noplane = outPath + "/show/noplane";
        std::string colorPath = ROOT_DICT + "/" + dataset.rgbList[index];
        cv::Mat colorMat = cv::imread(colorPath);
        cv::imwrite(out_path_noplane + "/" + cfg.output_head_name + "_" + std::to_string(index) + "_noplane.jpg", colorMat);
    }
}


int main(int argc, char** argv)
{
    /// Parser Arguments
    args_parser(argc,argv);

    config cfg;
    std::string cfgPath = ROOT_DICT + "/cfg.json";
    cfg.read(cfgPath);

    std::string jsonName = ROOT_DICT + "/" + fileName;
    TUMReader dataset;
    dataset.read_from_json(jsonName);
    int fileNum = dataset.depthList.size();
    std::cout << "Data in Total: " << fileNum << std::endl;

    /// NDT RANSAC for tum
    int index = 0 + start_index;
    int hasplane = 0;
    int plane_avg = 0;
    while(index < fileNum)
    {
        std::string depthPath = ROOT_DICT + "/" + dataset.depthList[index];
        cv::Mat depthMat = cv::imread(depthPath, cv::IMREAD_ANYDEPTH);
        PointCloud::Ptr cloud = d2cloud(depthMat, dataset.intrinsic, dataset.factor);
        int planeNum = 15;
        std::vector<PLANE> planes = ndtRANSAC(cloud, cfg, planeNum);
        outputResults(cfg, dataset, planes, index);
        std::cout << "\r" << "[" << index+1 <<  "/" << fileNum << "]" << std::flush;
        if(planes.size()>0){
            hasplane++;
            plane_avg += planes.size();
        }
        index ++ ;
    }
    double avg = double(plane_avg)/double(hasplane);
    std::cout << std::endl;
    std::cout << hasplane << " of " << fileNum << " has planes, average plane num: "
              << avg << std::endl;
    return (0);
}