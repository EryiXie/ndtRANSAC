#ifndef FUNCS_H
#define FUNCS_H

#include <Eigen/Dense>


static inline bool SortEigenValuesAndVectors(Eigen::Matrix3f& eigenVectors, Eigen::Vector3f& eigenValues)
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

static double compute_d(Eigen::Vector3f center1, Eigen::Vector3f center2,  Eigen::Vector3f normal1)
{
    return std::fabs((center1-center2).dot(normal1));
}

static double compute_thelta(Eigen::Vector3f normal1, Eigen::Vector3f normal2)
{
    return std::acos(std::fabs(normal1.dot(normal2)));
}


#endif