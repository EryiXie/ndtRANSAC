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


/*

    int c[] = {0, 1750, 3341, 3903, 3933, 3945, 3963, 7388, 7582, 7601, 7607, 8344, 9362};

    std::vector<int> colors(c, c+13);

    cv::Mat imgD = cv::Mat::zeros(img.size(), CV_32S);
    for (int n=0; n<img.rows; n++){
        for(int m=0; m<img.cols; m++){
            imgD.at<int>(n,m) = (img.at<cv::Vec3b>(n,m)[2] *256*256) + (img.at<cv::Vec3b>(n,m)[1] *256) + img.at<cv::Vec3b>(n,m)[0];
        }
    }

    for (int n=0; n<img.rows; n++){
        for(int m=0; m<img.cols; m++){
            if( img.at<cv::Vec3b>(n,m)[2] == 0 && img.at<cv::Vec3b>(n,m)[1] == 13 && img.at<cv::Vec3b>(n,m)[0] == 13 ) {
                std::cout << "find color gray." << std::endl;
                break;
            }
            else{
                
            }
        }
    }

    cv::Mat allmask = cv::Mat::zeros(img.size(), CV_8U);
    for (int j=0; j<colors.size(); j++)
    {
        std::cout << j << ", " << colors[j] << std::endl;
        cv::Mat mask = (imgD == colors[j]);
        cv::imwrite(std::to_string(j) + ".png", mask);
        allmask = allmask + mask;
    }

    cv::imwrite("all.png", allmask);
*/
    return 0 ;
}