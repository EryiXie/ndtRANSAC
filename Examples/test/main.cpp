#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


int main()
{
    cv::Mat img = cv::imread("/media/xie/4A6238FC6238EDF7/dataSet/datasetStanford/area_3/data/semantic/camera_f0e54fcd44df46cea3ac3bd97eab0bef_WC_1_frame_4_domain_semantic.png",
    -1);

    std::vector<std::vector<int>> colors;

    for (int n=0; n<img.rows; n++){
        for(int m=0; m<img.cols; m++){

            std::vector<int> color(3);
            color[0] = img.at<cv::Vec3b>(n,m)[0];  
            color[1] = img.at<cv::Vec3b>(n,m)[1] ;
            color[2] = img.at<cv::Vec3b>(n,m)[2];
            if( std::find(colors.begin(), colors.end(), color) != colors.end() ) {
            }
            else{
                colors.push_back(color);
            }
        }
    }

    for(unsigned int j=0; j<colors.size(); j++){
        std::cout << colors[j][0] <<"," << colors[j][1] << "," << colors[j][2] << std::endl;
    }
    
    return 0 ;
}