
#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include "cmdline.h"
#include "NdtOctree.h"
#include "visualizer.h"
#include "utils.h"

std::string ROOT_DICT;
std::string outPath = "";
std::string fileName = "color_map.json"; // can also use not mandotory parser
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


int main(int argc, char** argv)
{
    /// Parser Arguments
    args_parser(argc,argv);

    /// Read Point Cloud
    std::string jsonName = ROOT_DICT+"/"+fileName;
    std::vector<std::vector<std::string>> files = readJSON(jsonName);
    int fileNum = files.size();
    std::cout << "Data in Total: " << fileNum << std::endl;

    /// NDT RANSAC
    int index = 0;

    int image_id = 100000 + start_index;

    while(image_id-100000 < 60){

        std::string out_path_rgb = outPath + "/rgb_out1";
        std::string output_name_head = "bp" + std::to_string(image_id);
        cv::Mat colorMat = cv::imread(files[image_id-100000][1]);
        /// Visualize Result
        cv::resize(colorMat, colorMat, cv::Size(550,550));
        cv::imwrite(out_path_rgb + "/" + output_name_head + ".jpg", colorMat);
        image_id ++;
    }

    return (0);
}