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
std::string fileName = "tum.json";

int start_index;
int end_index;

void args_parser(int argc, char**argv)
{
    cmdline::parser arg;
    arg.add<std::string>("root", 'r', "root dict", true, "");
    arg.add<std::string>("outpath",'o',"output path",false,"");
    arg.add<int>("start",'s',"start index",false,0);
    arg.add<int>("end",'e',"end index",false,0);
    arg.parse_check(argc,argv);

    start_index = arg.get<int>("start");
    ROOT_DICT = arg.get<std::string>("root");
    outPath = arg.get<std::string>("outpath");
    if(outPath == "") outPath = ROOT_DICT;

}


std::vector<PLANE> ndtRANSAC(const PointCloud::Ptr &cloud, config cfg, unsigned int max_plane_per_cloud)
{
    std::vector<PLANE> planes;
    /// Establish Octree and Rough NDT-Segmentation
    NdtOctree ndtoctree();
    ndtoctree.setInputCloud(cloud, cfg.resolution);
    ndtoctree.computeLeafsNormal();
    ndtoctree.planarSegment(cfg.threshold);
    /// NDT-RANSAC
    ndtoctree.ndtRansac(planes, max_plane_per_cloud, cfg.delta_d, cfg.delta_thelta);
    return planes;
}

std::vector<PLANE> RANSAC(const PointCloud::Ptr &cloud, config cfg, unsigned int max_plane_per_cloud)
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
        seg.setDistanceThreshold (0.02);//cfg.delta_d);
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
            for(unsigned int i=0; i<cloud_inlier->points.size(); i++){
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

PLANE RANSAC_simple(const PointCloud::Ptr &cloud, config cfg, int max_plane_per_cloud)
{
    PLANE plane;
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
        seg.setDistanceThreshold (0.02);//cfg.delta_d);
        seg.setInputCloud (cloud_);
        seg.segment (*inliers, *coefficients);

        PointCloud::Ptr cloud_inlier (new pcl::PointCloud<PointT>);
        PointCloud::Ptr cloud_outlier (new pcl::PointCloud<PointT>);
        pcl::ExtractIndices<PointT> extract;
        extract.setInputCloud (cloud);
        extract.setIndices (inliers);
        extract.filter (*cloud_inlier);

        if(cloud_inlier->points.size() > cloud->points.size()*0.5){
            for(unsigned int i=0; i<cloud_inlier->points.size(); i++){
                plane.points.push_back(cloud_inlier->points[i]);
            }
            IRLS_plane_fitting(plane);
            break;
        }
        index ++;
    }
    return plane;
}


void outputResults(config cfg, TUMReader dataset, std::vector<PLANE> planes, int index, int image_id)
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
                cv::imwrite(out_path_masks + "/" + cfg.output_head_name + "_"  + std::to_string(image_id)
                            + "_plane_" + std::to_string(i) +".png", masks[i]);
        }

        if(cfg.use_present_sample)// && index%50 == 0)
        {
            mkdir(const_cast<char *>(out_path_show.c_str()), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
            cv::Mat colorMat = cv::imread(colorPath);
            cv::Mat show =  vs.take3in1_tum(masks, colorMat);
            cv::imwrite(out_path_show + "/" + cfg.output_head_name + "_" + std::to_string(image_id) + "_show.png", show);
        }



        if(cfg.use_output_resize)
        {
            mkdir(const_cast<char *>(out_path_dep.c_str()), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
            mkdir(const_cast<char *>(out_path_rgb.c_str()), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
            cv::Mat colorMat = cv::imread(colorPath);
            cv::resize(colorMat, colorMat, cv::Size(550,550));
            cv::Mat depMat = cv::imread(depthPath, cv::IMREAD_ANYDEPTH);
            cv::resize(depMat, depMat, cv::Size(550,550));
            cv::imwrite(out_path_rgb + "/" + cfg.output_head_name + "_" + std::to_string(image_id) + ".png", colorMat);
            cv::imwrite(out_path_dep + "/" + cfg.output_head_name + "_" + std::to_string(image_id) + "_dep.png", depMat);
        }

    }
    else{
        std::string out_path_noplane = outPath + "/show/noplane";
        std::string colorPath = ROOT_DICT + "/" + dataset.rgbList[index];
        cv::Mat colorMat = cv::imread(colorPath);
        cv::imwrite(out_path_noplane + "/" + cfg.output_head_name + "_" + std::to_string(image_id) + "_noplane.jpg", colorMat);
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

        int image_id;
        std::vector<std::string> dummy;
        std::vector<std::string> dummy2;
        split_string(dataset.depthList[index], '_', dummy);
        //std::cout << dummy[0] << std::endl;
        split_string(dummy[0], 'l', dummy2);
        image_id = std::stoi(dummy2[1]);
        //std::cout << image_id << std::endl;


        for (unsigned int i=0;i<dataset.maskList[index].size();i++)
        {

            cv::Mat mask = cv::imread(ROOT_DICT + "/" + dataset.maskList[index][i], 0);
            cv::Mat maskedDepthMat;
            depthMat.copyTo(maskedDepthMat, mask);
            PointCloud::Ptr cloud = d2cloud(maskedDepthMat, dataset.intrinsic, dataset.factor);
            int planeNum = 2;
            if(!cloud->points.empty()){
                //std::vector<PLANE> planes_per_semantic = RANSAC(cloud,cfg,planeNum);
                //if(!planes_per_semantic.empty())
                    //planes.insert(planes.end(),planes_per_semantic.begin(),planes_per_semantic.end());
                    planes.push_back(RANSAC_simple(cloud,cfg,planeNum));
            }
        }

        std::cout <<"potential planes: " << dataset.maskList[index].size() << std::flush;
        std::cout  << " before combine: " << planes.size() <<std::flush;
        //if(!planes.empty())
            //combine_planes(planes, planes, cfg.delta_d, cfg.delta_thelta);
        //std::cout << ", after combine: " << planes.size();

        // Do refinement on remained depth map
        visualizer vs;
        cv::Mat mask_all = vs.projectPlane2Mat(planes[0], dataset.intrinsic);
        for (unsigned int i = 1; i < planes.size(); i++) {
            mask_all += vs.projectPlane2Mat(planes[i], dataset.intrinsic);
        }
        cv::Mat mask_all_inv;
        cv::threshold(mask_all,mask_all_inv,1,255,cv::THRESH_BINARY_INV);
        cv::Mat maskedDepthMat_remained;
        depthMat.copyTo(maskedDepthMat_remained, mask_all_inv);
        PointCloud::Ptr cloud_re = d2cloud(maskedDepthMat_remained, dataset.intrinsic, dataset.factor);
        if(!cloud_re->points.empty()){
            NdtOctree ndtoctree();
            ndtoctree.setInputCloud(cloud_re, cfg.resolution);
            ndtoctree.computeLeafsNormal();
            ndtoctree.planarSegment(cfg.threshold);
            ndtoctree.refine_new(planes, cfg.delta_d, cfg.delta_thelta);
        }

        for (unsigned int i=0; i<planes.size(); i++)
            IRLS_plane_fitting(planes[i]);
        combine_planes(planes,planes,cfg.delta_d/2, cfg.delta_thelta/2);
        std::cout << ", after recombine: " << planes.size();
        /// ndtransac on rest points (find new planes)
        cv::Mat all_re = vs.projectPlane2Mat(planes[0], dataset.intrinsic);
        for (unsigned int i = 1; i < planes.size(); i++) {
            all_re += vs.projectPlane2Mat(planes[i], dataset.intrinsic);
        }
        cv::threshold(all_re,all_re,1,255,cv::THRESH_BINARY_INV);
        cv::Mat maskedDepthMat;
        depthMat.copyTo(maskedDepthMat, all_re);
        PointCloud::Ptr cloud_re_1 = d2cloud(maskedDepthMat, dataset.intrinsic, dataset.factor);
        if(cloud_re_1->points.size() > depthMat.cols*depthMat.rows*0.005){
            int max_remain_planes = 3;
            std::vector<PLANE> remain_planes = ndtRANSAC(cloud_re_1, cfg, max_remain_planes);
            for (unsigned int i=0; i<remain_planes.size(); i++)
                IRLS_plane_fitting(remain_planes[i]);
            combine_planes(remain_planes, remain_planes, cfg.delta_d, cfg.delta_thelta);
            planes.insert(planes.end(), remain_planes.begin(), remain_planes.end());
        }
        std::cout << ", after ndt and recombine: " << planes.size() << std::endl;
        std::vector<PLANE> plane_output;
        for(unsigned int i=0;i<planes.size();i++){
            if (planes[i].points.size() > depthMat.cols*depthMat.rows*0.005)
                plane_output.push_back(planes[i]);
        }


        std::sort(plane_output.begin(), plane_output.end(),
                  [](const PLANE & a, const PLANE & b){ return a.points.size() > b.points.size(); });

        outputResults(cfg, dataset, plane_output, index, image_id);
        std::cout << "\r" << "[" << index+1 <<  "/" << fileNum << "]" << std::endl << std::endl;;
        index ++ ;
    }
    return (0);
}

/// 目前的ndt方法，在工厂场景下会严重失效。
/// 考虑用ransac或者更稳定的ndt方法代替
/// 使用ransac的方法，输出并没有明显地更加理想
