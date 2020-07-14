#ifndef UTILS_H
#define UTILS_H

#include <sys/time.h>
#include <fstream>
#include <nlohmann/json.hpp>

//milisecond time stample
//use for measure the duration
//example: double t = tick(); ......; t = tick()-t;
double tick(void)
{
    struct timeval t;
    gettimeofday(&t, 0);
    return t.tv_sec + 1E-6 * t.tv_usec;
}

// split std::string to substring form char delim
// input: const std::string &str, char delim
// output: std::vector<std::string> &elems
std::vector<std::string> &split_string(const std::string &str, char delim, std::vector<std::string> &elems, bool skip_empty = true)
{
    std::istringstream iss(str);
    for (std::string item; getline(iss, item, delim); )
        if (skip_empty && item.empty()) continue;
        else elems.push_back(item);
    return elems;
}

std::string getKernelName(std::string &name)
{
    std::vector<std::string> strs1;
    split_string(name,'/',strs1);
    std::vector<std::string> strs2;
    split_string(strs1[strs1.size()-1],'_',strs2);

    std::string output = "";
    for(int i=0;i<strs2.size()-1;i++){
        output = output + strs2[i] + "_";
    }
    return output;
}

// read path for pose.json, rgb, depth and semantic images.
std::vector<std::vector<std::string>> readFileDict(std::string &path)
{
    std::ifstream ifs(path+"/sample_list_02.txt");
    std::vector<std::vector<std::string>> files_list;
    std::string str;
    int i = 0;
    std::vector<std::string> line(4);
    while(ifs >> str){
        if(i%4 == 0)
            line[0] = str;
        else if(i%4 == 1)
            line[1] = str;
        else if(i%4 == 2)
            line[2] = str;
        else{
            line[3] = str;
            files_list.push_back(line);
        }
        i++;
    }
    ifs.close();
    return files_list;
}

inline bool SortEigenValuesAndVectors(Eigen::Matrix3f& eigenVectors, Eigen::Vector3f& eigenValues)
{
    if (eigenVectors.cols() < 2 || eigenVectors.cols() != eigenValues.rows())
    {
        assert(false);
        return false;
    }

    unsigned n = eigenVectors.cols();
    for (unsigned i = 0; i < n - 1; i++)
    {
        unsigned maxValIndex = i;
        for (unsigned j = i + 1; j<n; j++)
            if (eigenValues[j] > eigenValues[maxValIndex])
                maxValIndex = j;

        if (maxValIndex != i)
        {
            std::swap(eigenValues[i], eigenValues[maxValIndex]);
            for (unsigned j = 0; j < n; ++j)
                std::swap(eigenVectors(j, i), eigenVectors(j, maxValIndex));
        }
    }

    return true;
}


#endif //NDT_UTILS_H