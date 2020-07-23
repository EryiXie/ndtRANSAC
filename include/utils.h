#ifndef UTILS_H
#define UTILS_H

#include <sys/time.h>
#include <fstream>
#include <nlohmann/json.hpp>

//milisecond time stample
//use for measure the duration
//example: double t = tick(); ......; t = tick()-t;
static double tick(void)
{
    struct timeval t;
    gettimeofday(&t, 0);
    return t.tv_sec + 1E-6 * t.tv_usec;
}

// split std::string to substring form char delim
// input: const std::string &str, char delim
// output: std::vector<std::string> &elems
static std::vector<std::string> &split_string(const std::string &str, char delim, std::vector<std::string> &elems, bool skip_empty = true)
{
    std::istringstream iss(str);
    for (std::string item; getline(iss, item, delim); )
        if (skip_empty && item.empty()) continue;
        else elems.push_back(item);
    return elems;
}

static std::vector<int> vecstr_to_vecint(std::vector<std::string> vs)
{
    std::vector<int> ret;
    for(std::vector<std::string>::iterator it=vs.begin();it!=vs.end();++it)
    {
        std::istringstream iss(*it);
        int temp;
        iss >> temp;
        ret.push_back(temp);
    }  
    return ret;
}

#endif
