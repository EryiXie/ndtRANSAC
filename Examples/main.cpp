#inculde <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include "cmdline.h"
#include "NdtOctree.h"
#include "visualizer.h"
#include "utils.h"

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloud;

double resolution = 0.05;
double threshold = 0.02;
double delta_d = 0.05;
double delta_thelta = 15.0;
std::string dictPath;
std::string outPath;
std::string fileName;
int start_index;

double t1,t2,t3,t4,t5,t6;

void args_parser(int argc, char**argv)
{
    cmdline::parser arg;
    arg.add<std::string>("inpath",'i',"input path of files dictionary",true,"");
    arg.add<std::string>("outpath",'o',"output path",true,"");
    arg.add<std::string>("file",'f',"json file",true,"");
    arg.add<int>("start",'s',"start index",true,0);

    arg.parse_check(argc,argv);

    dictPath = arg.get<std::string>("inpath");
    outPath = arg.get<std::string>("outpath") + "/";
    fileName = arg.get<std::string>("file");
    start_index = arg.get<int>("start");
}

PointCloud::Ptr d2cloud_with_semantic(const cv::Mat &depth, std::vector<cv::Point> semantic,
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

double computeloss(PLANE plane){
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

std::vector<std::vector<cv::Point>> readMasks(std::string &semantic_path, std::string &colormap_path)
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
            split(colormap_string[i],' ', color_strings);
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

void ndtRANSAC_rgbd(std::vector<std::string> &names,
                    const std::string &output_path,
                    int image_id)
{
    std::string pose_path = names[0];
    std::string rgb_path = names[1];
    std::string depth_path= names[2];
    std::string semantic_path= names[3];

    std::string output_name_head = "pb" + std::to_string(image_id);

    /// Read Pose and Depth,then Generalize Point Cloud
    visualizer vs;
    Eigen::Matrix3f intrinsics_matrix = vs.readIntrinsic(pose_path);
    cv::Mat depthMat = cv::imread(depth_path,cv::IMREAD_ANYDEPTH);
    /// Read Semantic and Generalize Indices
    std::string colormap_path = dictPath + "/" + fileName;
    std::vector<std::vector<cv::Point>> semantic_maps = readMasks(semantic_path,colormap_path);

    std::vector<PLANE> planes;
    for(int i=0;i<semantic_maps.size();i++)
    {
        if(semantic_maps[i].size()< depthMat.cols*depthMat.rows*0.01) continue;
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

    std::cout <<"planes of img: " << planes.size() << std::endl;

    std::sort(planes.begin(), planes.end(),
              [](const PLANE & a, const PLANE & b){ return a.points.size() > b.points.size(); });

    /// Visulize Result
    std::vector<cv::Mat> masks;
    std::vector<double> losses;
    int planeNum = std::min(int(planes.size()),10);
    for(int i=0;i<planeNum;i++){
        cv::Mat mask = vs.projectPlane2Mat(planes[i],intrinsics_matrix);
        masks.push_back(mask);
        losses.push_back(computeloss(planes[i]));
    }

    cv::Mat colorMat = cv::imread(rgb_path);
    cv::Mat total_mask = vs.maskSuperpositon(masks);
    cv::Mat masked = vs.take3in1(masks,colorMat,losses);

    cv::imwrite(output_path +"/rgb_out_irls/" + output_name_head + ".jpg", colorMat);
    cv::imwrite(output_path +"/masked_irls/" + output_name_head + "masked.png",masked);
}



int main(int argc, char** argv)
{
    /// Parser Arguments
    args_parser(argc,argv);

    /// Read Point Cloud
    std::string jsonName = "/"+fileName;
    std::cout << jsonName << std::endl;
    std::vector<std::vector<std::string>> files = readJSON(dictPath, jsonName);
    size_t fileNum = files.size();
    std::cout << "Data in Total: " << fileNum << std::endl;

    /// NDT RANSAC
    int index = 0+start_index;
    int image_id = 100000 + index;
    while(index<fileNum)
    {
        t1 = tick();
        std::cout << "img: " << files[index][0] << std::endl;
        image_id ++;
        ndtRANSAC_rgbd(files[index],outPath,image_id);
        t1 = tick()-t1;
        std::cout << "Processed image[" << index << "], " << "consumed time: " << t1 << " seconds" << std::endl;
        index ++;
    }

    return (0);
}