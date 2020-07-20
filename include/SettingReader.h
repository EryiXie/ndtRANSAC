//
// Created by eryi on 24.09.19.
//

#ifndef SETTINGREADER_H
#define SETTINGREADER_H

#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <Eigen/Dense>
//#include "utils.h"

class config
{
public:

    static double resolution;
    static double threshold;
    static double delta_d;
    static double delta_thelta;

    static int max_output_planes;
    static std::string output_head_name;
    static bool use_semantic_mask;
    static bool use_output_resize;
    static bool use_present_sample;
    static bool use_indiv_masks;

    config();
    void read(std::string cfgPath);

private:
};

class DatasetReader{
public:
    static double factor;
    static Eigen::Matrix3f intrinsic;

};

class TUMReader: public DatasetReader{

public:
    static std::vector<std::string> depthList;
    static std::vector<std::string> rgbList;
    static std::vector<std::vector<std::string>> maskList;

    void help();

    void read_from_json(std::string &jsonName);
};

class BPReader: public DatasetReader{
public:
  
    void help();

    std::vector<std::vector<std::string>> readJSON(std::string &jsonName);

    Eigen::Matrix3f readIntrinsic( std::string &jsonName);

};

class NYUReader: public DatasetReader{
public:
    static std::vector<std::string> depthList;
    static std::vector<std::string> rgbList;
    static std::vector<std::vector<std::string>> maskList;

    void help();

    void read_from_json(std::string &jsonName);

};


#endif
