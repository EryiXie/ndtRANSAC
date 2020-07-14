#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include "cmdline.h"
#include "NdtOctree_re.h"
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
std::string fileName = "tumval_new.json";

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

std::vector<PLANE> RANSAC(const PointCloud::Ptr &cloud, config cfg, int max_plane_per_cloud)
{
    std::vector<PLANE> planes;
    int index = 0;
    while(index < 50){
        PointCloud::Ptr cloud_ (new pcl::PointCloud<PointT>);
        pcl::copyPointCloud(*cloud, *cloud_);
        pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
        pcl::SACSegmentation<PointT> seg;
        seg.setOptimizeCoefficients (true);
        seg.setModelType (pcl::SACMODEL_PLANE);
        seg.setMethodType (pcl::SAC_RANSAC);
        seg.setDistanceThreshold (cfg.delta_d);
        seg.setInputCloud (cloud_);
        seg.segment (*inliers, *coefficients);

        PointCloud::Ptr cloud_inlier (new pcl::PointCloud<PointT>);
        PointCloud::Ptr cloud_outlier (new pcl::PointCloud<PointT>);
        pcl::ExtractIndices<PointT> extract;
        extract.setInputCloud (cloud);
        extract.setIndices (inliers);
        extract.filter (*cloud_inlier);

        if(cloud_inlier->points.size() > cloud->points.size()*0.1){
            PLANE plane;
            for(int i=0; i<cloud_inlier->points.size(); i++){
                plane.points.push_back(cloud_inlier->points[i]);
            }
            IRLS_plane_fitting(plane);
            planes.push_back(plane);
        }
        extract.setNegative (true);
        extract.filter (*cloud_outlier);
        pcl::copyPointCloud(*cloud_outlier, *cloud_);
        if (cloud_outlier->points.size() < cloud->points.size()*0.1 || planes.size() > max_plane_per_cloud){
            break;
        }
        index ++;
    }
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
            cv::imwrite(out_path_show + "/" + cfg.output_head_name + "_" + std::to_string(10000+index) + "_show.png", show);
        }
        if(cfg.use_output_resize)
        {
            mkdir(const_cast<char *>(out_path_dep.c_str()), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
            mkdir(const_cast<char *>(out_path_rgb.c_str()), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
            cv::Mat colorMat = cv::imread(colorPath);
            cv::resize(colorMat, colorMat, cv::Size(550,550));
            cv::Mat depMat = cv::imread(depthPath, cv::IMREAD_ANYDEPTH);
            cv::resize(depMat, depMat, cv::Size(550,550));
            cv::imwrite(out_path_rgb + "/" + cfg.output_head_name + "_" + std::to_string(10000+index) + ".png", colorMat);
            cv::imwrite(out_path_dep + "/" + cfg.output_head_name + "_" + std::to_string(10000+index) + "_dep.png", depMat);
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
    while(index < fileNum)
    {
        std::string depthPath = ROOT_DICT + "/" + dataset.depthList[index];
        std::cout << depthPath << std::endl;
        cv::Mat depthMat = cv::imread(depthPath, cv::IMREAD_ANYDEPTH);
        std::vector<PLANE> planes;

        for (int i=0;i<dataset.maskList[index].size();i++)
        {

            cv::Mat mask = cv::imread(ROOT_DICT + "/" + dataset.maskList[index][i], 0);
            cv::Mat maskedDepthMat;
            depthMat.copyTo(maskedDepthMat, mask);
            PointCloud::Ptr cloud = d2cloud(maskedDepthMat, dataset.intrinsic, dataset.factor);
            int planeNum = 2;
            if(!cloud->points.empty()){
                std::vector<PLANE> planes_per_semantic = RANSAC(cloud,cfg,planeNum);
                if(!planes_per_semantic.empty())
                    planes.insert(planes.end(),planes_per_semantic.begin(),planes_per_semantic.end());
            }
        }

        std::cout << std::endl <<"potential planes: " << dataset.maskList[index].size() << std::flush;
        std::cout  << " before combine: " << planes.size() <<std::flush;
        if(!planes.empty())
            combine_planes(planes, planes, cfg.delta_d, cfg.delta_thelta);
        std::sort(planes.begin(), planes.end(),
                  [](const PLANE & a, const PLANE & b){ return a.points.size() > b.points.size(); });

        std::cout << ", after combine: " << planes.size() << std::endl;

        for(int i=0;i<planes.size();i++){
            //planes[i].points.clear();
            planes[i].points.shrink_to_fit();
        }


        // Do refine on whole depth map
        PointCloud::Ptr cloud = d2cloud(depthMat, dataset.intrinsic, dataset.factor);

        NdtOctree ndtoctree(cfg.resolution);
        ndtoctree.setInputCloud(cloud);
        ndtoctree.computeLeafsNormal();
        ndtoctree.planarSegment(cfg.threshold);
        ndtoctree.refine_new(planes, cfg.delta_d, cfg.delta_thelta);

        for (int i=0; i<planes.size(); i++)
            IRLS_plane_fitting(planes[i]);
        combine_planes(planes,planes,cfg.delta_d, cfg.delta_thelta);

        /// ndtransac on rest points
        visualizer vs;
        std::vector<cv::Mat> masks;
        cv::Mat all = vs.projectPlane2Mat(planes[0], dataset.intrinsic);
        for (int i = 1; i < planes.size(); i++) {
            all += vs.projectPlane2Mat(planes[i], dataset.intrinsic);
        }
        cv::bitwise_not(all, all);
        cv::threshold(all,all,1,255,cv::THRESH_BINARY);

        bool mark = cv::sum(all)[0]/255 > (depthMat.cols*depthMat.rows*0.4);
        std::cout << mark << " " << sum(all)[0]/255 << " "<< (depthMat.cols*depthMat.rows*0.4) << std::endl;
        if(mark){
            cv::Mat maskedDepthMat_re;
            depthMat.copyTo(maskedDepthMat_re, all);
            PointCloud::Ptr cloud_re = d2cloud(maskedDepthMat_re, dataset.intrinsic, dataset.factor);
            int max_remain_planes = 3;
            std::vector<PLANE> remain_planes = ndtRANSAC(cloud_re,cfg,max_remain_planes);
            for(int i=0;i<remain_planes.size();i++){
                //remain_planes[i].points.clear();
                remain_planes[i].points.shrink_to_fit();
            }
            //NdtOctree ndtoctree_re(cfg.resolution);
            //ndtoctree_re.setInputCloud(cloud_re);
            //ndtoctree_re.computeLeafsNormal();
            //ndtoctree_re.planarSegment(cfg.threshold);
            //ndtoctree_re.refine_new(remain_planes, cfg.delta_d, cfg.delta_thelta);
            //planes.insert(planes.end(), remain_planes.begin(), remain_planes.end());
        }

        for (int i=0; i<planes.size(); i++)
            IRLS_plane_fitting(planes[i]);
        combine_planes(planes,planes,cfg.delta_d, cfg.delta_thelta);

        outputResults(cfg, dataset, planes, index);
        // 先通过yolact给出的分析，得到一系列的高质量平面，是否使用ndt还是使用普通ransac呢？ 
        // 对平面进行查重，然后以这些平面为输入，用整个图来refine这些平面，生成的新Plane全体，求此外的反区域，再执行一次ndtRansac
        // 第一阶段是完成： 得到高质量平面，查重，refine，可视化

        std::cout << "\r" << "[" << index+1 <<  "/" << fileNum << "]" << std::flush;
        index ++ ;
    }
    return (0);
}