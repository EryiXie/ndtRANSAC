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

// TUM RGB-D has such a low quality depth data, that we can't extract solid plane regions from it correctly, even using ndt ransac.
// Here we use the prediction of PlaneRCNN as a prior, first detect planes inside the plane regions predicted by PlaneRCNN,
// Then we perfrom ndt ransac on the remain regions.


typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloud;

std::string ROOT_DICT;
std::string outPath = "";
std::string fileName = "tum(copy).json";
std::string cfgName = "cfg.json";

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
    PointCloud::Ptr outliers(new PointCloud);
    ndtoctree.ndtRansac(planes, outliers, max_plane_per_cloud, cfg.delta_d, cfg.delta_thelta);
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
    TUMReader TUM;
    TUM.read_from_json(datasetPath);
    int fileNum = TUM.depthList.size();
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
        std::string depthPath = ROOT_DICT + "/" + TUM.depthList[index];
        std::cout << depthPath << std::endl;
        cv::Mat depthMat = cv::imread(depthPath, cv::IMREAD_ANYDEPTH);
        std::vector<PLANE> planes;

        std::string image_id;
        std::vector<std::string> dummy;
        std::vector<std::string> dummy2;
        split_string(TUM.rgbList[index], '/', dummy);
        split_string(dummy[1], '.', dummy2);
        image_id = dummy2[0];
        

        for (unsigned int i=0;i<TUM.priorLists[index].size();i++)
        {
            cv::Mat mask = cv::imread(ROOT_DICT + "/" + TUM.priorLists[index][i], cv::IMREAD_GRAYSCALE);
            cv::Mat maskedDepthMat;
            depthMat.copyTo(maskedDepthMat, mask);
            PointCloud::Ptr cloud = d2cloud(maskedDepthMat, TUM.intrinsic, TUM.factor);
            int planeNum = 2;
            if(!cloud->points.empty()){
                std::vector<PLANE> tempPlanes = RANSAC(cloud, cfg, planeNum);
                planes.insert(planes.end(), tempPlanes.begin(), tempPlanes.end());
            }
        }

        std::cout <<"potential planes: " << TUM.priorLists[index].size() << std::flush;
        std::cout  << " before combine: " << planes.size() <<std::flush;

       // Do refinement on remained depth map
        visualizer vs(depthMat.size());
        cv::Mat mask_all = vs.projectPlane2Mat(planes[0], TUM.intrinsic);
        for (unsigned int i = 1; i < planes.size(); i++) {
            mask_all += vs.projectPlane2Mat(planes[i], TUM.intrinsic);
        }

        cv::Mat mask_all_inv;
        cv::threshold(mask_all,mask_all_inv,1,255,cv::THRESH_BINARY_INV);
        cv::Mat maskedDepthMat_remained;
        depthMat.copyTo(maskedDepthMat_remained, mask_all_inv);
        PointCloud::Ptr cloud_re = d2cloud(maskedDepthMat_remained, TUM.intrinsic, TUM.factor);
     
        if(!cloud_re->points.empty()){
            std::vector<PLANE> ndt_planes;
            int max_plane_per_cloud = 3;
            NdtOctree ndtoctree;
            ndtoctree.setInputCloud(cloud_re, cfg.resolution);
            ndtoctree.computeLeafsNormal();
            ndtoctree.planarSegment(cfg.threshold);
            PointCloud::Ptr outliers(new PointCloud);
            ndtoctree.ndtRansac(ndt_planes, outliers, max_plane_per_cloud, cfg.delta_d*1.25, cfg.delta_thelta*1.25);
            planes.insert(planes.end(), ndt_planes.begin(), ndt_planes.end());
        }

        // Do refinement on remained depth map
        cv::Mat mask_all2 = vs.projectPlane2Mat(planes[0], TUM.intrinsic);
        for (unsigned int i = 1; i < planes.size(); i++) {
            mask_all2 += vs.projectPlane2Mat(planes[i], TUM.intrinsic);
        }
        
        cv::Mat mask_all_inv2;
        cv::threshold(mask_all2,mask_all_inv2,1,255,cv::THRESH_BINARY_INV);
        cv::Mat maskedDepthMat_remained2;
        depthMat.copyTo(maskedDepthMat_remained2, mask_all_inv2);
        PointCloud::Ptr cloud_re2 = d2cloud(maskedDepthMat_remained2, TUM.intrinsic, TUM.factor);
        refine_planes_with_remainpoints(planes,cloud_re2,cfg.delta_d*1.25, cfg.delta_thelta*1.25);

        for (unsigned int i=0; i<planes.size(); i++)
            planes[i].IRLS_paras_fitting();
        combine_planes(planes,planes,cfg.delta_d, cfg.delta_thelta);
        std::cout << ", after recombine: " << planes.size();

        std::vector<PLANE> plane_output;
        for(unsigned int i=0;i<planes.size();i++){
            if (planes[i].points.size() > depthMat.cols*depthMat.rows*0.05)
                plane_output.push_back(planes[i]);
        }

        std::sort(plane_output.begin(), plane_output.end(),
                  [](const PLANE & a, const PLANE & b){ return a.points.size() > b.points.size(); });

        std::cout << "final " << plane_output.size() << std::endl;

        outputResults(cfg,depthMat.size(), TUM, plane_output, index, image_id);
        std::cout << "\r" << "[" << index+1 <<  "/" << fileNum << "]" << std::endl << std::endl;;
        index ++ ;
    }
    return (0);
}
