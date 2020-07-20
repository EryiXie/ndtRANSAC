#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include <cmdline.h>
#include "NdtOctree.h"
#include "visualizer.h"
#include "utils.h"
#include "SettingReader.h"
#include "funcs.h"
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
std::string fileName = "nyu.json";
std::string cfgName = "cfg_nyu.json";

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
    end_index = arg.get<int>("end");
    ROOT_DICT = arg.get<std::string>("root");
    outPath = arg.get<std::string>("outpath");
    if(outPath == "") outPath = ROOT_DICT;

}

std::vector<PLANE> ndtRANSAC(const PointCloud::Ptr &cloud, config cfg, unsigned int max_plane_per_cloud)
{
    std::vector<PLANE> planes;
    /// Establish Octree and Rough NDT-Segmentation
    NdtOctree ndtoctree;
    ndtoctree.setInputCloud(cloud, cfg.resolution);
    ndtoctree.computeLeafsNormal();
    ndtoctree.planarSegment(cfg.threshold);
    /// NDT-RANSAC
    ndtoctree.ndtRansac(planes, max_plane_per_cloud, cfg.delta_d, cfg.delta_thelta);

    return planes;
}

std::vector<PLANE> RANSAC(const PointCloud::Ptr &cloud, config cfg, int max_plane_per_cloud)
{
    std::vector<PLANE> planes;
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
    pcl::SACSegmentation<PointT> seg;
    pcl::ExtractIndices<PointT> extract;
    seg.setOptimizeCoefficients (true);
    seg.setModelType (pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setDistanceThreshold (cfg.delta_d);

    PointCloud::Ptr could_pub(new pcl::PointCloud<PointT>);
    unsigned int original_size = cloud->points.size();
    int n_planes = 0;
    while(cloud->points.size() > original_size * 0.1)
    {
        // Fit a plane
        seg.setInputCloud(cloud);
        seg.segment(*inliers, *coefficients);

        // Check result
        if (inliers->indices.size() == 0)
            break;
        
        if (inliers->indices.size() > original_size * 0.1)
        {
            PLANE plane;
            for (unsigned int i=0;i<inliers->indices.size();i++)
                plane.points.push_back(cloud->points[inliers->indices[i]]);
            plane.IRLS_paras_fitting();
            planes.push_back(plane);
            n_planes++;
        }
        // Extract inliers
        extract.setInputCloud(cloud);
        extract.setIndices(inliers);
        extract.setNegative(true);
        PointCloud cloudF;
        extract.filter(cloudF);
        cloud->swap(cloudF);

        if (n_planes >= max_plane_per_cloud) break;
    }
    return planes;
}

void outputResults(config cfg, TUMReader dataset, std::vector<PLANE> planes, int index, std::string image_id)
{
    if(planes.size()>0) {
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
                cv::imwrite(out_path_masks + "/" + image_id
                            + "_plane_" + std::to_string(i) +".png", masks[i]);
        }

        if(cfg.use_present_sample)// && index%50 == 0)
        {
            mkdir(const_cast<char *>(out_path_show.c_str()), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
            cv::Mat colorMat = cv::imread(colorPath);
            cv::Mat show =  vs.take3in1_tum(masks, colorMat);
            cv::imwrite(out_path_show + "/" + image_id + "_show.png", show);
        }

        if(cfg.use_output_resize)
        {
            mkdir(const_cast<char *>(out_path_dep.c_str()), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
            mkdir(const_cast<char *>(out_path_rgb.c_str()), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
            cv::Mat colorMat = cv::imread(colorPath);
            cv::resize(colorMat, colorMat, cv::Size(550,550));
            cv::Mat depMat = cv::imread(depthPath, cv::IMREAD_ANYDEPTH);
            cv::resize(depMat, depMat, cv::Size(550,550));
            cv::imwrite(out_path_rgb + "/" + image_id + ".png", colorMat);
            cv::imwrite(out_path_dep + "/" + image_id + "_dep.png", depMat);
        }
    }
    else{
        std::string out_path_noplane = outPath + "/show/noplane";
        std::string colorPath = ROOT_DICT + "/" + dataset.rgbList[index];
        cv::Mat colorMat = cv::imread(colorPath);
        cv::imwrite(out_path_noplane + "/" + image_id + ".png", colorMat);
    }
}

int main(int argc, char** argv)
{
    /// Parser Arguments
    args_parser(argc,argv);
    config cfg;
    std::string cfgPath = ROOT_DICT + "/" + cfgName;
    cfg.read(cfgPath);

    std::string datasetPath = ROOT_DICT + "/" + fileName;
    NYUReader dataset;
    dataset.read_from_json(datasetPath);
    int fileNum = dataset.depthList.size();
    std::cout << "Data in Total: " << fileNum << std::endl;

    int index = 0 + start_index;
    int end_at;
    if (end_index > start_index || end_index > 0 )
        end_at = end_index;
    else
        end_at = fileNum;
    
    //main loop
    while(index < end_at)
    {
        std::string depthPath = ROOT_DICT + "/" + dataset.depthList[index];
        std::string semanticPath = ROOT_DICT + "/" + dataset.semanticList[index];
        std::cout << depthPath << std::endl;
        cv::Mat depthMat = cv::imread(depthPath, cv::IMREAD_ANYDEPTH);
        cv::Mat semanticMat = cv::imread(semanticPath, cv::IMREAD_UNCHANGED);
        std::vector<PLANE> planes;
        std::string image_id;
        std::vector<std::string> dummy;
        std::vector<std::string> dummy2;
        split_string(dataset.rgbList[index], '/', dummy);
        split_string(dummy[1], '.', dummy2);
        image_id = dummy2[0];
        std::cout << image_id << std::endl;

        std::vector<PLANE> planes;
        for(unsigned int i=0;i<dataset.colorList[index].size();i++){
            cv::Mat mask = cv::Mat::ones(maskMat.size(), CV_8UC1) * dataset.colorList[index][i];
            cv::Mat semanticMask = cv::Mat::zeros(maskMat.size(), CV_16UC1);
            cv::Mat maskedDepthMat;
            cv::bitwise_xor(semanticMat, mask, semanticMask);
            semanticMask = semanticMask*2;
            cv::threshold(semanticMask, semanticMask, 1, 255, CV_THRESH_BINARY_INV);
            depthMat.copyTo(maskedDepthMat,semanticMask);
            //cv::imwrite(ROOT_DICT + "/" + std::to_string(i) + ".png", maskedDepthMat);

            PointCloud::Ptr cloud = d2cloud(maskedDepthMat, dataset.intrinsic, dataset.factor);
            int planeNum = 3;
            if(!cloud->points.empty()){
                std::vector<PLANE> tempPlanes = RANSAC(cloud, cfg, planeNum);
                planes.insert(planes.end(), tempPlanes.begin(), tempPlanes.end());
            }
        }

        std::cout <<"potential planes: " << dataset.maskList[index].size() << std::flush;
        std::cout  << " before combine: " << planes.size() <<std::flush;

        
        index ++;
    }
    return (0);
}
