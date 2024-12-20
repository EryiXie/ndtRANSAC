#ifndef NDTOCTREE_H
#define NDTOCTREE_H

#define _USE_MATH_DEFINES

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/octree/octree_search.h>
#include <pcl/octree/octree_base.h>
#include <pcl/octree/octree_nodes.h>

#include <pcl/features/normal_3d.h>
#include <pcl/common/impl/centroid.hpp>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <cmath>
#include <algorithm>
#include <random>

#include "utils.h"
#include "funcs.h"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloud;

class PLANE
{
public:
    std::vector<PointT> points;
    Eigen::Vector3f normal;
    Eigen::Vector3f center;

    PLANE()
    {

    }
    double computeloss()
    {
        double sum=0;
        for(unsigned int i=0;i<points.size();i++){
            Eigen::Vector3f pt;
            pt << points[i].x,points[i].y,points[i].z;
            float dotproduct = std::fabs((center-pt).dot(normal));
            sum = sum + dotproduct;
        }
        double loss = sum/points.size()*1000;

        return loss;
    }

    void IRLS_paras_fitting()
    {
        int max_iterations_ = 1000;
        int min_iterations = 2;
        double threshold2stop = 1e-5;

        if (points.size()>0)
        {
            unsigned num_of_points = points.size();
            Eigen::Vector3f meanSum_;
            Eigen::Matrix3f covSum_;

            Eigen::Vector3f mean_;
            mean_<< 0,0,0;
            Eigen::Matrix3f cov_;
            Eigen::Matrix3f evecs_;
            Eigen::Vector3f evals_;

            mean_<<0,0,0;
            for(unsigned int i=0; i< num_of_points; i++)
            {
                Eigen::Vector3f tmp;
                tmp<<points[i].x,points[i].y,points[i].z;
                mean_ += tmp;
            }
            meanSum_ = mean_;
            mean_ /= (num_of_points);
            Eigen::MatrixXf mp;
            mp.resize(num_of_points,3);
            for(unsigned int i=0; i< num_of_points; i++)
            {
                mp(i,0) = points[i].x - mean_(0);
                mp(i,1) = points[i].y - mean_(1);
                mp(i,2) = points[i].z - mean_(2);
            }
            covSum_ = mp.transpose()*mp;
            cov_ = covSum_/(num_of_points-1);
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> Sol (cov_);

            evecs_ = Sol.eigenvectors().real();
            evals_ = Sol.eigenvalues().real();
            SortEigenValuesAndVectors(evecs_, evals_);

            Eigen::Vector3f normal_;
            normal_[0]=evecs_(0,2);//其它的Cell
            normal_[1]=evecs_(1,2);
            normal_[2]=evecs_(2,2);

            normal << normal_[0], normal_[1], normal_[2];
            center = mean_;

            //进行Robust平面获取
            Eigen::Vector3f oldnormal=normal_;
            //IRLS主循环
            for (int iter=0;iter<max_iterations_;iter++)
            {
                oldnormal=normal_;
                float  sum_dist=0;
                //循环1，求距离均值
                for (unsigned i=0; i<num_of_points; ++i)
                {
                    const pcl::PointXYZ CellP=points[i];
                    Eigen::Vector3f pt(CellP.x,CellP.y,CellP.z);
                    float distance=fabs(normal.dot(pt-mean_));//点到平面的距离
                    sum_dist+=distance;
                }
                //距离均值
                double mean_dist_=sum_dist/num_of_points;
                double s_dist_mean=0;
                //循环2.求距离方差
                for (unsigned i=0; i<num_of_points; ++i)
                {
                    const pcl::PointXYZ CellP=points[i];
                    Eigen::Vector3f pt(CellP.x,CellP.y,CellP.z);
                    float distance=fabs(normal_.dot(pt-mean_));//点到平面的距离
                    s_dist_mean+=(distance-mean_dist_)*(distance-mean_dist_);
                }
                //距离方差
                double sigma=sqrt(s_dist_mean/(num_of_points-1));
                if (sigma<0.000001)
                {
                    break;
                }
                //循环3.过滤满足小于2*sigma的点，根据这些点求新的均值
                //求距离均值
                int num_of_ok_points=0;
                Eigen::Vector3f mean_new;
                Eigen::Matrix3f covSum_new;

                mean_new<<0,0,0;
                for (unsigned i=0; i<num_of_points; ++i)
                {
                    const pcl::PointXYZ CellP=points[i];
                    Eigen::Vector3f pt(CellP.x,CellP.y,CellP.z);
                    float distance=fabs(normal_.dot(pt-mean_));//点到平面的距离
                    if (distance-mean_dist_<sigma*2)
                    {
                        //记录该点索引
                        num_of_ok_points++;
                        mean_new+=pt;
                    }
                }
                mean_new/=num_of_ok_points;
                //根据标记的点重新计算均值和方差，产生新的均值点和法向量
                //计算新的方差
                Eigen::MatrixXf mp_1;
                mp_1.resize(num_of_ok_points,3);
                int indx_of_ok_point=0;
                for (unsigned i=0; i<num_of_points; ++i)
                {
                    const pcl::PointXYZ CellP=points[i];
                    Eigen::Vector3f pt(CellP.x,CellP.y,CellP.z);
                    float distance=fabs(normal_.dot(pt-mean_));//点到平面的距离
                    if (distance-mean_dist_<sigma*2)
                    {
                        //记录该点索引
                        mp_1(indx_of_ok_point,0) = points[i].x - mean_new(0);
                        mp_1(indx_of_ok_point,1) = points[i].y - mean_new(1);
                        mp_1(indx_of_ok_point,2) = points[i].z - mean_new(2);
                        indx_of_ok_point++;
                    }
                }
                covSum_new = mp_1.transpose()*mp_1;
                cov_ = covSum_new/(num_of_ok_points-1);
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> Sol_1 (cov_);

                evecs_ = Sol_1.eigenvectors().real();
                evals_ = Sol_1.eigenvalues().real();
                SortEigenValuesAndVectors(evecs_, evals_);

                normal_[0]=evecs_(0,2);
                normal_[1]=evecs_(1,2);
                normal_[2]=evecs_(2,2);
                mean_= mean_new;
                //判断是否终止循环
                Eigen::Vector3f temp=oldnormal-normal_;
                double* con=new double[3];
                con[0]=std::abs(temp[0])/std::abs(oldnormal[0]);
                con[1]=std::abs(temp[1])/std::abs(oldnormal[1]);
                con[2]=std::abs(temp[2])/std::abs(oldnormal[2]);
                std::sort(con,con+3);
                double	convg = con[2];
                if (iter>=min_iterations)
                {
                    if (convg < threshold2stop)
                    {
                        break;
                    }
                }
            }
            //结束循环，获取了优化的均值点和法向量
            normal << normal_[0], normal_[1], normal_[2];
            center = mean_;
        }
    }

    void PCA_plane_fitting()
    {
        PointCloud::Ptr plane_cloud (new PointCloud);
        plane_cloud->points.assign(points.begin(),points.end());
        Eigen::Vector4f temp_center;
        pcl::compute3DCentroid(*plane_cloud,temp_center);
        center << temp_center(0), temp_center(1),temp_center(2);
        pcl::NormalEstimation<PointT,pcl::Normal> ne;
        float nx,ny,nz, curvature;
        std::vector<int> fake_indices(points.size());
        std::iota(fake_indices.begin(),fake_indices.end(),0);
        ne.computePointNormal(*plane_cloud,fake_indices,nx,ny,nz,curvature);
        normal << nx,ny,nz;
    }
};

class NdtOctree {

    struct LEAF {
        std::vector<int> indices;
        std::vector<PointT> points; // Maybe no need.
        Eigen::Matrix3f covariance_;
        Eigen::Vector3f centroid_;
        Eigen::Vector3f eigvals_;
        float curvature_;
    };

public:
    //Constructor
    NdtOctree(){}

    // Empty deconstructor
    ~NdtOctree(){}

    void setInputCloud(const PointCloud::Ptr &cloud, double min_resolution)
    {
        min_resolution_ = min_resolution;
        pcl::octree::OctreePointCloudSearch<PointT>::AlignedPointTVector voxel_centers;
        pcl::octree::OctreePointCloudSearch<PointT> octree (min_resolution_);
        cloud_ = cloud;
        octree.setInputCloud(cloud_);
        octree.addPointsFromInputCloud();
        /// Reasonable resolution need to be reconsidered
        //double volume = octree.getLeafCount()*std::pow(min_resolution_,3);
        //double reso1 = std::pow(volume/1000.0,1.0/3.0);

        //reso1 = std::max(reso1, 0.01);

        //octree.deleteTree();
        //octree.setResolution(reso1);
        //octree.setResolution();
        //octree.addPointsFromInputCloud();
        octree.getOccupiedVoxelCenters(voxel_centers); // get a PointT vector of all occupied voxels.
        treeDepth_ = octree.getTreeDepth();
        leafsNum_ = octree.getLeafCount();

        leafs.resize(voxel_centers.size());
        for(size_t i=0;i<leafsNum_;i++){
            octree.voxelSearch(voxel_centers[i],leafs[i].indices);
            for(size_t j=0;j<leafs[i].indices.size();j++)
                leafs[i].points.push_back(cloud_->points[leafs[i].indices[j]]);
        }
    }

    void computeLeafsNormal()
    {
        for(size_t i=0;i<leafsNum_;i++)
        {
            Eigen::Vector4f temp_center;
            pcl::compute3DCentroid(*cloud_,leafs[i].indices,temp_center);
            leafs[i].centroid_ << temp_center(0), temp_center(1),temp_center(2);
            // extract points from point cloud belong to leafs[i]
            if(leafs[i].indices.size()<3){
                leafs[i].covariance_ << 0,0,0,0,0,0,0,0,0;
                leafs[i].eigvals_ << 0,0,0;
                leafs[i].curvature_ = -1.0f;
                // if less than 3 point: set curvature to minus value, and eigenvals to zero.
            }
            else{
                pcl::NormalEstimation<PointT,pcl::Normal> ne;
                float nx,ny,nz;
                ne.computePointNormal(*cloud_,leafs[i].indices,nx,ny,nz,leafs[i].curvature_);
                leafs[i].eigvals_ << nx,ny,nz;
            }
        }
    }

    void planarSegment(const double threshold=0.04)
    {
        for(size_t i=0;i<leafs.size();i++)
        {
            if( !leafs[i].eigvals_.isZero() && leafs[i].curvature_ < threshold){
                leafDict_planar.push_back(i);
                for(unsigned int j=0;j<leafs[i].indices.size();j++)
                    pointDict_planar.push_back(leafs[i].indices[j]);
            }
            else{
                leafDict_non.push_back(i);
                for(unsigned int j=0;j<leafs[i].indices.size();j++)
                    pointDict_non.push_back(leafs[i].indices[j]);
            }
        }
        is_planeSeged = true;
    }

    void ndtRansac(std::vector<PLANE> &planes, PointCloud::Ptr &outliers,
                   const int maxPlaneNum = 3,
                   const double delta_d = 0.05,
                   const double delta_thelta = 20/180.0*M_PI)
    {
        if(leafDict_planar.size() < 100){
            return;
        }

        std::vector<PLANE> outputs;
        std::vector<int> cellList;//this list will be reassigned by doRansac_on_leafs()
        cellList.assign(leafDict_planar.begin(),leafDict_planar.end());

        int n=0;
        unsigned int max_remainNum = std::max(int(leafDict_planar.size()*0.05), 1);
        while(n < maxPlaneNum)
        {
            PLANE temp_plane = doRansac_on_leafs(cellList, delta_d, delta_thelta);
            if(temp_plane.points.empty())
                break;
            else if(temp_plane.points.size() > cloud_->points.size()*0.05) // 意义不明确
                outputs.push_back(temp_plane);

            unsigned int remainNum = cellList.size();
            if(remainNum < max_remainNum)
                break;
            n++;
        }
        std::vector<int> remainList;
        remainList.assign(pointDict_non.begin(),pointDict_non.end());

        for(unsigned int i=0;i<cellList.size();i++){
            int index = cellList[i];
            remainList.insert(remainList.end(),leafs[index].indices.begin(),leafs[index].indices.end());
        }

        for(unsigned int i=0;i<remainList.size();i++)
            outliers->points.push_back(cloud_->points[remainList[i]]);

        planes.assign(outputs.begin(),outputs.end());
    }

    void refine_planes_with_ndtvoxel(std::vector<PLANE> &planes,
            const double delta_d = 0.05,
            const double delta_thelta = 20/180.0*M_PI)
    {
        if(planes.empty()) return;

        std::vector<LEAF> leafs_planar;
        for(unsigned int j=0;j<leafDict_planar.size();j++)
        {
            leafs_planar.push_back(leafs[leafDict_planar[j]]);
        }

        for(unsigned int i=0;i<leafs_planar.size();i++)
        {
            LEAF leaf = leafs[i];

            std::vector<double> distances;
            std::vector<double> angles;
            std::vector<double> universal_factors;
            for(unsigned int j=0;j<planes.size();j++)
            {
                double d = compute_d(planes[j].center, leaf.centroid_, planes[j].normal);
                distances.push_back(d);
                double t = compute_thelta(planes[j].normal, leaf.eigvals_);
                angles.push_back(t);
            }
            double D = *max_element(distances.begin(),distances.end());
            double T = *max_element(angles.begin(), angles.end());
            for(unsigned int k=0; k<planes.size();k++){
                    universal_factors.push_back(distances[k]*distances[k]/(D*D) + angles[k]*angles[k]/(T*T));
            }
            auto it = min_element(std::begin(universal_factors), std::end(universal_factors));
            int plane_bestMatch = std::distance(std::begin(universal_factors), it);
            if(distances[plane_bestMatch]< delta_d && angles[plane_bestMatch]< delta_thelta){
                planes[plane_bestMatch].points.insert(planes[plane_bestMatch].points.end(),leaf.points.begin(),leaf.points.end());
            }
        }
    }

    void report()
    {
        if(cloud_  && !leafs.empty()){
            std::cout << std::endl << "***************************************************" << std::endl;
            std::cout << "Octree Established. " << std::endl
                      << "Leafs: " << leafsNum_ << "    "
                      << "min_resolution: " << min_resolution_*100 << " cm" << "     "
                      << "Tree Depth: " << treeDepth_ << std::endl;
            if (is_planeSeged){
                std::cout << std::endl <<  "Rough Plane Segmentation excuted." << std::endl
                          << "Co-Planar Leafs: " << leafDict_planar.size() <<  "     "
                          << "Non-Plnar Leafs: " << leafDict_non.size() << std::endl;
            }
            std::cout << "***************************************************" << std::endl << std::endl ;
        }
        else{

        }
    }

    float getmin_resolution(){return min_resolution_;}
    int getTreeDepth() {return treeDepth_;}
    size_t getLeafCount(){return leafsNum_;}

private:
    double min_resolution_; // the size of octree voxel.
    int treeDepth_;
    size_t leafsNum_;
    bool is_planeSeged = false;

    PointCloud::Ptr cloud_;
    std::vector<int> pointDict_planar;
    std::vector<int> pointDict_non;

    std::vector<LEAF> leafs;
    std::vector<int> leafDict_planar;
    std::vector<int> leafDict_non;

    void sort_vec(const Eigen::Vector3f& vec, Eigen::Vector3f& sorted_vec)
    {
        Eigen::Vector3i ind=Eigen::Vector3i::LinSpaced(vec.size(),0,vec.size()-1);
        //[0 1 2 3 ... N-1]
        auto rule=[vec](int i, int j)->bool{ return vec(i)<vec(j); };
        //正则表达式，作为sort的谓词
        std::sort(ind.data(),ind.data()+ind.size(),rule); //data成员函数返回VectorXd的第一个元素的指针，类似于begin()
        sorted_vec.resize(vec.size());
        for(unsigned int i=0;i<vec.size();i++){
            sorted_vec(i)=vec(ind(i));
        }
    }

    PLANE doRansac_on_leafs(std::vector<int> &leafDict_in, 
                            const double delta_d = 0.05,
                            const double delta_thelta = 20/180.0*M_PI)
    {
        size_t k_max = 200; // pow(0.95, 100) = 0.6%
        if(leafDict_in.size() <= k_max){
            k_max = leafDict_in.size();
        }
        size_t k = 0;
        std::vector<int> planeInliersDict;
        int leafsNum = leafDict_in.size();

        std::random_device rd;
        std::mt19937 engine(rd());
        std::uniform_int_distribution<int> dist(0,leafsNum-1);

        auto I_kmax = leafDict_in.size() * 0.1;
        while (k < k_max){
            int randcellIndex = dist(engine);
            std::vector<int> I_k;
            LEAF sample_leaf = leafs[leafDict_in[randcellIndex]];
            for(size_t i=0;i<leafDict_in.size();i++){
                double d = compute_d(sample_leaf.centroid_, leafs[leafDict_in[i]].centroid_ , sample_leaf.eigvals_);
                double thelta = compute_thelta(leafs[leafDict_in[i]].eigvals_, sample_leaf.eigvals_);
                if(d < delta_d && thelta < delta_thelta)
                    I_k.push_back(leafDict_in[i]);
            }
            if(I_k.size() > I_kmax){
                I_kmax = I_k.size();
                double pn = double(I_k.size()) / double(leafDict_in.size());
                k_max = int (log(1-0.99)/log(1-pn));
                planeInliersDict.assign(I_k.begin(),I_k.end());
            }
            k++;
        }

        //Earse extracted from leafDict_in
        for(unsigned int i=0;i<planeInliersDict.size();i++){
            std::vector<int>::iterator it;
            it = std::find(leafDict_in.begin(),leafDict_in.end(),planeInliersDict[i]);
            leafDict_in.erase(it);
        }

        //Output Plane
        PLANE output;
        for (unsigned int i=0;i<planeInliersDict.size();i++){
            LEAF leaf = leafs[planeInliersDict[i]];
            output.points.insert(output.points.end(),leaf.points.begin(),leaf.points.end());
        }
        output.IRLS_paras_fitting();
        return output;        
    }
};

static __attribute__((unused)) void combine_planes(std::vector<PLANE> &src, 
                            std::vector<PLANE> &dst,
                            double delta_d=0.05, 
                            double delta_thelta=20/180.0*M_PI)
{
    if (src.empty()) return;
    dst.assign(src.begin(),src.end());
    for (unsigned int i=0;i<dst.size()-1;i++){
        for (unsigned int j=i+1;j<dst.size();j++){
            double d = compute_d(dst[i].center,dst[j].center,dst[i].normal);
            double thelta = compute_thelta(dst[i].normal,dst[j].normal);
            if(d < delta_d && thelta < delta_thelta){
                dst[i].points.insert(dst[i].points.end(), dst[j].points.begin(), dst[j].points.end());
                dst.erase(dst.begin()+j);
                dst[i].IRLS_paras_fitting();
                j = j-1;
            }
        }
    }
}

static __attribute__((unused)) void refine_planes_with_remainpoints(std::vector<PLANE> &planes,
                                            const PointCloud::Ptr &outliers,
                                            const double delta_d = 0.05,
                                            const double delta_thelta = 20/180.0*M_PI)
    {
        if(planes.empty()  || outliers->points.empty()) return;
        ulong knb = 20;
        pcl::KdTreeFLANN<PointT> kd;
        kd.setInputCloud(outliers);
        std::vector<int> neighbors(knb);
        std::vector<float> neighborsdistances(knb);
        pcl::NormalEstimation<PointT,pcl::Normal> ne;

        for (unsigned int i=0; i<outliers->points.size(); i++){
            Eigen::Vector3f pt_center;
            pt_center << outliers->points[i].x, outliers->points[i].y, outliers->points[i].z;
            std::vector<double> distances;
            for(unsigned int j=0;j<planes.size();j++){
                double d = compute_d(planes[j].center,pt_center,planes[j].normal);
                distances.push_back(d);
            }
            auto it = min_element(std::begin(distances), std::end(distances));
            int plane_bestMatch = std::distance(std::begin(distances), it);
            if (distances[plane_bestMatch] < delta_d){
                kd.nearestKSearch(i,int(knb),neighbors,neighborsdistances);
                float nx,ny,nz,curvature;
                ne.computePointNormal(*outliers,neighbors,nx,ny,nz,curvature);
                Eigen::Vector3f pt_normal;
                pt_normal << nx,ny,nz;
                double thelta = compute_thelta(pt_normal, planes[plane_bestMatch].normal);
                if (thelta < delta_thelta)  planes[plane_bestMatch].points.push_back(outliers->points[i]);
            }
        }
    }

static __attribute__((unused))PointCloud::Ptr d2cloud(cv::Mat &depth,
                        Eigen::Matrix3f &intrinsics_matrix,
                        double &factor)
{
    // Convert depth image into colorized 3d point cloud
    double const fx_d = intrinsics_matrix(0,0);
    double const fy_d = intrinsics_matrix(1,1);
    double const cx_d = intrinsics_matrix(0,2);
    double const cy_d = intrinsics_matrix(1,2);

    PointCloud::Ptr cloud(new PointCloud);

    int num = 0;
    for(int m=0;m<depth.rows;m++){
        for(int n=0; n<depth.cols;n++){
            float d = depth.at<ushort>(m,n);
            
            if(d==0 || d>65535*0.75) {
                //do something
            }
            else{
                PointT p;
                num ++;
                // calculate xyz coordinate with camera intrinsics paras
                p.z = float (d/factor);
                p.x = float ((n - cx_d) * p.z / fx_d);
                p.y = float ((m - cy_d) * p.z / fy_d);
                cloud->points.push_back(p);
            }
        }
    }
    cloud->width = cloud->points.size();
    cloud->height = 1;
    return cloud;
}

#endif
