//
// Created by eryi on 24.09.19.
//

#ifndef NDT_SETTINGREADER_H
#define NDT_SETTINGREADER_H

#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>

class config
{
public:

    double resolution;
    double threshold;
    double delta_d;
    double delta_thelta;

    int max_output_planes;
    std::string output_head_name;
    bool use_semantic_mask;
    bool use_output_resize;
    bool use_present_sample;
    bool use_indiv_masks;

    config(){
        resolution = 0.05;
        threshold = 0.02;
        delta_d = 0.05;
        delta_thelta = 15.0;

        max_output_planes = 15;
        use_semantic_mask = false;
        use_present_sample = true;
        use_output_resize = false;
        use_indiv_masks = false;
        output_head_name = "";
    }

    ~config(){}

    void read(std::string cfgPath)
    {
        std::ifstream file(cfgPath);
        nlohmann::json js;
        if(file.good()){
            file >> js;
            resolution = js["resolution"].get<double>();
            threshold = js["threshold"].get<double>();
            delta_d =js["delta_d"].get<double>();
            delta_thelta = js["delta_thelta"].get<double>();
            max_output_planes = js["max_output_planes"].get<int>();
            output_head_name = js["output_head_name"].get<std::string>();
            use_semantic_mask = js["use_semantic_mask"].get<bool>();
            use_output_resize = js["use_output_resize"].get<bool>();
            use_present_sample = js["use_present_sample"].get<bool>();
            use_indiv_masks = js["use_indiv_masks"].get<bool>();
        }
        else{
            std::cout << "json file not found." << std::endl;
        }
    }

private:


};

class DatasetReader{
public:
    double factor;
    Eigen::Matrix3f intrinsic;

};

class TUMReader: public DatasetReader{

public:
    std::vector<std::string> depthList;
    std::vector<std::string> rgbList;
    std::vector<std::vector<std::string>> maskList;

    void help()
    {
        // gives or reads message from the help data elements,
        // in order to help understand the data structure of the dataset related json files
    }


    void read_from_json(std::string &jsonName)
    {
        std::vector<std::vector<std::string>> files_list;

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

            for(int i=0;i<vecs.size();i++){
                depthList.push_back(vecs[i][1]);
                rgbList.push_back(vecs[i][0]);
                int maskcount = std::stoi(vecs[i][3]);
                std::vector<std::string> maskline;
                for(int j=0; j<= maskcount;j++)
                    maskline.push_back( vecs[i][2] + "_plane_"+ std::to_string(j) + ".png");
                maskList.push_back(maskline);
            }
        }
        else{
            std::cout << "json file not found." << std::endl;
        }
    }
};

class BPReader: public DatasetReader{
public:
    void help()
    {
        // gives or reads message from the help data elements,
        // in order to help understand the data structure of the dataset related json files
    }

    std::vector<std::vector<std::string>> readJSON(std::string &jsonName)
    {
        std::vector<std::string> foo;
        split_string(jsonName,'/',foo);
        std::string jsonPath = "";
        for(int i=0; i<foo.size()-1; i++) jsonPath = jsonPath + "/" + foo[i];
        std:: cout << jsonPath << ", " << jsonName << std::endl;
        std::vector<std::vector<std::string>> files_list;
        std::ifstream file(jsonName);
        nlohmann::json js;
        std::vector<std::string> vec;
        if (file.good()){
            file >> js;
            vec = js["img_list"].get<std::vector<std::string>>();
            for(int i=0;i<vec.size();i++){
                std::vector<std::string> line;
                split_string(vec[i],' ', line);
                for(int j=0;j<line.size();j++){
                    line[j] = jsonPath + line[j];
                }
                files_list.push_back(line);
            }
        }
        else{
            std::cout << "json file not found." << std::endl;
        }
        return files_list;
    }

    Eigen::Matrix3f readIntrinsic( std::string &jsonName)
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

};



#endif //NDT_SETTINGREADER_H
