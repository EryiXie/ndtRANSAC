//
// Created by eryi on 24.09.19.
//

#ifndef SETTINGREADER_H
#define SETTINGREADER_H

#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <Eigen/Dense>

class config
{
public:

    static double resolution; //resolution of ndt voxel (in meter)
    static double threshold; //threshold for planar voxels
    static double delta_d; //plane ransac threshold distance
    static double delta_thelta; //plane ransc threshold angular

    static int max_output_planes; //max number of extract planes
    static bool use_output_resize;
    static bool use_present_sample;
    static bool use_indiv_masks;
    static bool use_total_masks;
    config();
    void read(const std::string &cfgPath);
};

class DatasetReader{
public:
    static double factor;
    static Eigen::Matrix3f intrinsic;
    static std::vector<std::string> depthList;
    static std::vector<std::string> rgbList;
};


class TUMReader: public DatasetReader{

public:
    static std::vector<std::vector<std::string>> priorLists;
    void read_from_json(const std::string &jsonName);
};

class NYUReader: public DatasetReader{
public:
    static std::vector<std::string> maskList;
    static std::vector<std::vector<int>> labels;
    void read_from_json(const std::string &jsonName);
};


class BPReader: public DatasetReader{
public:
    static std::vector<std::string> poseList;
    static std::vector<std::string> maskList;
    static std::vector<std::vector<int>> labels;
    void read_from_json(const std::string &jsonName);
    Eigen::Matrix3f readIntrinsic(const std::string &jsonName);

};


#endif
