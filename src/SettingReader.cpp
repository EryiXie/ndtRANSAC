#include "SettingReader.h"
#include <fstream>
#include <iostream>
#include "utils.h"
#include <cmath>

double config::resolution;
double config::threshold;
double config::delta_d;
double config::delta_thelta;

int config::max_output_planes;
bool config::use_output_resize;
bool config::use_present_sample;
bool config::use_indiv_masks;
bool config::use_total_masks;

std::vector<std::string> DatasetReader::depthList;
std::vector<std::string> DatasetReader::rgbList;

std::vector<std::vector<std::string>> TUMReader::priorLists;
std::vector<std::string> NYUReader::maskList;
std::vector<std::vector<int>> BPReader::labels;
std::vector<std::string> BPReader::maskList;
std::vector<std::string> BPReader::poseList;
std::vector<std::vector<int>> NYUReader::labels;


double DatasetReader::factor;
Eigen::Matrix3f DatasetReader::intrinsic;


// configuration reader
config::config()
{
    resolution = 0.05;
    threshold = 0.02;
    delta_d = 0.05;
    delta_thelta = 15.0/180.0*M_PI;

    max_output_planes = 15;
    use_present_sample = true;
    use_output_resize = false;
    use_indiv_masks = false;
}

void config::read(const std::string &cfgPath)
{
    std::ifstream file(cfgPath);
    nlohmann::json js;
    if(file.good()){
        file >> js;
        resolution = js["resolution"].get<double>();
        threshold = js["threshold"].get<double>();
        delta_d =js["delta_d"].get<double>();
        delta_thelta = js["delta_thelta"].get<double>()/180.0*M_PI;
        max_output_planes = js["max_output_planes"].get<int>();
        use_output_resize = js["use_output_resize"].get<bool>();
        use_present_sample = js["use_present_sample"].get<bool>();
        use_indiv_masks = js["use_indiv_masks"].get<bool>();
        use_total_masks = js["use_total_masks"].get<bool>();
    }
    else{
        std::cout << "json file not found." << std::endl;
    }
}



// Dataset Readers

void TUMReader::read_from_json(const std::string &jsonName)
{
    std::ifstream file(jsonName);
    nlohmann::json js;
    std::vector<std::vector<std::string>> vecs;
    if(file.good()){
        file >> js;
        vecs = js["samples"].get<std::vector<std::vector<std::string>>>();
        factor = js["factor"].get<float>();
        intrinsic << js["intrinsic"][0], 0, js["intrinsic"][2],
                0, js["intrinsic"][1],js["intrinsic"][3],
                0,0,1;

        for(unsigned int i=0;i<vecs.size();i++)
        {  
            rgbList.push_back(vecs[i][0]);
            depthList.push_back(vecs[i][1]);
            int maskcount = std::stoi(vecs[i][3]);
            std::vector<std::string> maskline;
            for(int j=0; j< maskcount;j++)
                maskline.push_back(vecs[i][2] + "_plane_"+ std::to_string(j) + ".png");
           priorLists.push_back(maskline);
        }
    }
    else{
         std::cout << "json file not found." << std::endl;
    }
}

void BPReader::read_from_json(const std::string &jsonName)
{
    std::ifstream file(jsonName);
    nlohmann::json js;
    std::vector<std::vector<std::string>> vecs;
    if(file.good()){
        file >> js;
        vecs = js["samples"].get<std::vector<std::vector<std::string>>>();
        factor = js["factor"].get<float>();
        for(unsigned int i=0;i<vecs.size();i++)
        {  
            rgbList.push_back(vecs[i][0]);
            depthList.push_back(vecs[i][1]);
            poseList.push_back(vecs[i][2]);
            maskList.push_back(vecs[i][3]);
            std::vector<std::string> dummy;
            split_string(vecs[i][4], ',', dummy);
            labels.push_back(vecstr_to_vecint(dummy));
        }
    }
    else{
         std::cout << "json file not found." << std::endl;
    }
}

Eigen::Matrix3f BPReader::readIntrinsic(const std::string &jsonName)
{
    Eigen::Matrix3f intrinsic;

    std::ifstream file(jsonName);
    nlohmann::json j;
    if (file.good()) {
        file >> j;
        intrinsic << j["camera_k_matrix"][0][0], j["camera_k_matrix"][0][1], j["camera_k_matrix"][0][2],
                j["camera_k_matrix"][1][0], j["camera_k_matrix"][1][1], j["camera_k_matrix"][1][2],
                j["camera_k_matrix"][2][0], j["camera_k_matrix"][2][1], j["camera_k_matrix"][2][2];
    } else {
        intrinsic << 0, 0, 0, 0, 0, 0, 0, 0, 0;
        std::cerr << "Can't find camera pose related *.json file." << std::endl;
    }
    file.close();
    return intrinsic;
}

void NYUReader::read_from_json(const std::string &jsonName)
{
    std::ifstream file(jsonName);
    nlohmann::json js;
    std::vector<std::vector<std::string>> vecs;
    if(file.good()){
        file >> js;
        vecs = js["samples"].get<std::vector<std::vector<std::string>>>();
        factor = js["factor"].get<float>();
        intrinsic << js["intrinsic"][0], 0, js["intrinsic"][2],
                0, js["intrinsic"][1],js["intrinsic"][3],
                0,0,1;

        for(unsigned int i=0;i<vecs.size();i++)
        {
            rgbList.push_back(vecs[i][0]);
            depthList.push_back(vecs[i][1]);
            maskList.push_back(vecs[i][2]);
            std::vector<std::string> dummy;
            split_string(vecs[i][3], ',', dummy);
            labels.push_back(vecstr_to_vecint(dummy));
        }
    }
    else
    {
         std::cout << "json file not found." << std::endl;
    }
}