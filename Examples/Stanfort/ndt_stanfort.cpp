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
std::string fileName = "stanfort.json";
std::string cfgName = "cfg_stanfort.json";

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

std::vector<PLANE> ndtRANSAC(const PointCloud::Ptr &cloud, 
                            const config &cfg, 
                            const unsigned int &max_plane_per_cloud)
{
    std::vector<PLANE> planes;
    /// Establish Octree and Rough NDT-Segmentation
    NdtOctree ndtoctree;
    ndtoctree.setInputCloud(cloud, cfg.resolution);
    ndtoctree.computeLeafsNormal();
    ndtoctree.planarSegment(cfg.threshold);
    /// NDT-RANSAC
    PointCloud::Ptr outliers(new PointCloud);
    ndtoctree.ndtRansac(planes, outliers, max_plane_per_cloud, cfg.delta_d, cfg.delta_thelta);
    refine_planes_with_remainpoints(planes, outliers, cfg.delta_d, cfg.delta_thelta);
    for (unsigned int i=0; i<planes.size(); i++) 
        planes[i].IRLS_paras_fitting();
    combine_planes(planes,planes,cfg.delta_d, cfg.delta_thelta);

    return planes;
}

void outputResults(const config& cfg, 
                    const cv::Size& frameSize, 
                    const DatasetReader& dataset, 
                    const std::vector<PLANE>& planes, 
                    const int& index, 
                    const std::string& image_id)
{
    if(planes.size()>0) {
        visualizer vs(frameSize);
        std::vector<cv::Mat> masks;
        std::string out_path_rgb = outPath + "/rgb_out";
        std::string out_path_dep = outPath + "/dep_out";
        std::string out_path_masks = outPath + "/masks";
        std::string out_path_show = outPath + "/show";
        std::string out_path_one_mask = outPath + "/mask";
        std::string colorPath = ROOT_DICT + "/" + dataset.rgbList[index];

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

        if(cfg.use_total_masks){
            mkdir(const_cast<char *>(out_path_one_mask.c_str()), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
            cv::Mat grayscale_mask = cv::Mat::zeros(frameSize, CV_8UC1);
            for (int i = 0; i < planeNum; i++){
                int gray_scale = 32 + 8*i; // Support 29 plane instances from 32 to 255
                grayscale_mask = grayscale_mask + masks[i]/255*gray_scale;
            }
            cv::imwrite(out_path_one_mask + "/" + image_id + ".png", grayscale_mask);
        }

        if(cfg.use_present_sample && index%200 == 0)
        {
            mkdir(const_cast<char *>(out_path_show.c_str()), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
            cv::Mat colorMat = cv::imread(colorPath);
            cv::Mat show =  vs.draw_colormap_blend_labels(masks, colorMat);
            cv::imwrite(out_path_show + "/" + image_id + "_show.png", show);
        }
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
    BPReader Stanfort;
    
    Stanfort.read_from_json(datasetPath);
    int fileNum = Stanfort.depthList.size();
    std::cout << "Data in Total: " << fileNum << std::endl;

    int index = 0 + start_index;
    int end_at;
    if (end_index > start_index || end_index > 0 )
        end_at = end_index;
    else
        end_at = fileNum;

    while(index < end_at)
    {
        std::string depthPath = ROOT_DICT + "/" + Stanfort.depthList[index];
        std::string posePath = ROOT_DICT + "/" + Stanfort.poseList[index];
        std::string semanticPath = ROOT_DICT + "/" + Stanfort.maskList[index];

        cv::Mat depthMat = cv::imread(depthPath, cv::IMREAD_ANYDEPTH);
        cv::Mat semanticMask = cv::imread(ROOT_DICT + "/" + Stanfort.maskList[index], cv::IMREAD_UNCHANGED);
        Stanfort.intrinsic = Stanfort.readIntrinsic(posePath);

        std::vector<PLANE> planes;

        std::string image_id;
        std::vector<std::string> dummy;
        std::vector<std::string> dummy2;
        split_string(Stanfort.rgbList[index], '/', dummy);
        split_string(dummy[1], '.', dummy2);
        image_id = dummy2[0];

        cv::Mat semanticMask_32S = cv::Mat::zeros(semanticMask.size(), CV_32S);
        for (int n=0; n<semanticMask.rows; n++){
            for(int m=0; m<semanticMask.cols; m++){
                semanticMask_32S.at<int>(n,m) = (semanticMask.at<cv::Vec3b>(n,m)[2] *256*256) 
                + (semanticMask.at<cv::Vec3b>(n,m)[1] *256) + semanticMask.at<cv::Vec3b>(n,m)[0];
            }
        }

        cv::Mat filteredLabelMat_inv = cv::Mat::zeros(depthMat.size(), CV_8UC1);
        for (unsigned int j=0; j<Stanfort.labels[index].size(); j++) {
            cv::Mat mask = (semanticMask_32S == Stanfort.labels[index][j]);
            cv::Scalar sums = cv::sum(mask)/255;
            if(double(sums[0]) > semanticMask.cols*semanticMask.rows*0.005)
            {
                cv::Mat maskedDepthMat;
                filteredLabelMat_inv = filteredLabelMat_inv + mask;
                depthMat.copyTo(maskedDepthMat, mask);
                PointCloud::Ptr cloud = d2cloud(maskedDepthMat, Stanfort.intrinsic, Stanfort.factor);
                int planeNum = 2;
                if(!cloud->points.empty()){
                    std::vector<PLANE> tempPlanes = ndtRANSAC(cloud, cfg, planeNum);
                    planes.insert(planes.end(), tempPlanes.begin(), tempPlanes.end());
                }
            }
        }

       cv::threshold(filteredLabelMat_inv,filteredLabelMat_inv,1,255,cv::THRESH_BINARY_INV);

        // Do refinement on remained depth map
        visualizer vs(depthMat.size());
        cv::Mat mask_all_inv = vs.projectPlane2Mat(planes[0], Stanfort.intrinsic);
        for (unsigned int i = 1; i < planes.size(); i++) 
            mask_all_inv += vs.projectPlane2Mat(planes[i], Stanfort.intrinsic);
        cv::threshold(mask_all_inv,mask_all_inv,1,255,cv::THRESH_BINARY_INV);

        cv::Mat maskedDepthMat_remained;
        depthMat.copyTo(maskedDepthMat_remained, mask_all_inv);
        maskedDepthMat_remained.copyTo(maskedDepthMat_remained, filteredLabelMat_inv);
        PointCloud::Ptr cloud_re = d2cloud(maskedDepthMat_remained, Stanfort.intrinsic, Stanfort.factor);
        if(!cloud_re->points.empty()){
            NdtOctree ndtoctree;
            ndtoctree.setInputCloud(cloud_re, cfg.resolution);
            ndtoctree.computeLeafsNormal();
            ndtoctree.planarSegment(cfg.threshold);
            ndtoctree.refine_planes_with_ndtvoxel(planes, cfg.delta_d*0.25, cfg.delta_thelta*0.25);
        }
        for (unsigned int i=0; i<planes.size(); i++) planes[i].IRLS_paras_fitting();
        combine_planes(planes,planes,cfg.delta_d*0.5, cfg.delta_thelta*0.5);
      
        std::vector<PLANE> plane_output;
        for(unsigned int i=0;i<planes.size();i++){
            if (planes[i].points.size() > depthMat.cols*depthMat.rows*0.01)
                plane_output.push_back(planes[i]);
        }

        std::sort(plane_output.begin(), plane_output.end(),
                  [](const PLANE & a, const PLANE & b){ return a.points.size() > b.points.size(); });

        outputResults(cfg,depthMat.size(), Stanfort, plane_output, index, image_id);
        std::string outline = "\r[" + std::to_string(index+1) +  "/" + std::to_string(fileNum) + "]\n";
        std::cout << outline;

        index ++ ;
    }
}