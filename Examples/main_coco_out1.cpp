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

std::vector<std::vector<cv::Point>> readMasks(std::string &semantic_path, 
                                              std::string &colormap_path)
{
    cv::Mat semanticMat = cv::imread(semantic_path);
    std::ifstream file(colormap_path);
    std::vector<cv::Scalar> colormap;

    int pos = semantic_path.find_last_of('/');
    std::string semantic_name = semantic_path.substr(pos+1);

    std::vector<std::string> colormap_string;
    nlohmann::json j;
    if (file.good()){
        file >> j;
        colormap_string = j[semantic_name].get<std::vector<std::string>>();
        for(int i=0;i<colormap_string.size();i++){
            std::vector<std::string> color_strings;
            split_string(colormap_string[i],' ', color_strings);
            int b = std::stoi(color_strings[0]);
            int g = std::stoi(color_strings[1]);
            int r = std::stoi(color_strings[2]);
            colormap.push_back(cv::Scalar(b,g,r));
        }
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
    std::vector<std::vector<cv::Point>> semantic_maps = readMasks(semantic_path,colormap_path);

    std::vector<PLANE> planes;
    for(int i=0;i<semantic_maps.size();i++)
    {
        if(semantic_maps[i].size()< depthMat.cols*depthMat.rows*0.005) {
            continue;
        }
        /*
        else if (semantic_maps[i].size() < depthMat.cols*depthMat.rows*0.05){
            PointCloud::Ptr cloud = d2cloud_with_semantic(depthMat,semantic_maps[i],intrinsics_matrix);


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
