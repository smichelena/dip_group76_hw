//============================================================================
// Name        : Dip1.cpp
// Author      : Ronny Haensch
// Version     : 2.0
// Copyright   : -
// Description : 
//============================================================================

#include "Dip1.h"

#include <stdexcept>
#include <math.h>

namespace dip1 {

//rotation transformation
std::vector<int> rotation(std::vector<int> input, float angle){
    const double sine = sin(3.14159265*(angle/180));
    const double cosine = cos(3.14159265*(angle/180));
    double x = (double) input[0];
    double y = (double) input[1];
    return {(int)(x*cosine + y*sine), (int)(-x*sine + y*cosine) };
}

/**
 * @brief function that performs some kind of (simple) image processing
 * @param img input image
 * @returns output image
 */
cv::Mat doSomethingThatMyTutorIsGonnaLike(const cv::Mat& img) {
    // TO DO !!!

    //get rotation angle from user (uncomment if you want to try more angles)
    // I am using a fixed angle of 15 degrees to rotate the image because 
    // e.g angle = 0 doesnt pass the test
    // float angle; 
    // std::cout << "enter an angle between 0 and 90 degrees" << std::endl;
    // std::cin >> angle;
    // while(angle < 0 || angle > 90){
    //     std::cout << "please enter a valid rotation angle" << std::endl;
    //     std::cin >> angle;
    // }

    double angle = 15.0;

    //for ease of calculation
    const double sine = sin(3.14159265*(angle/180));
    const double cosine = cos(3.14159265*(angle/180));

    //get new image dimensions and axis correction
    const int length = rotation({img.cols - 1, img.rows - 1}, angle)[0];
    const int x_correction = abs(rotation({0, img.rows - 1}, angle)[0]);

    // std::cout << " new length: " << length << std::endl;

    //create new triple channel image
    cv::Mat output = cv::Mat(length, length, CV_8UC3);

    //fill output image with black pixels
    for (int x = 0; x < length; x++){
        for (int y = 0; y < length; y++){
           output.at<cv::Vec3b>(x,y) = {0, 0, 0};
        }
    }

    //rotate image
    for (int x = 0; x < img.cols; x++){
        for (int y = 0; y < img.rows; y++){
           std::vector<int> current = rotation({y,x}, angle);
           //this abomination is to avoid segmentation faults
           if (current[0] >= 0 && current[1] + x_correction >= 0 && current[0] < output.cols && current[0] < output.rows &&
            current[1] + x_correction < output.cols && current[1] + x_correction < output.rows)
            //place pixel in axis corrected coordinate
           output.at<cv::Vec3b>(current[0], current[1] + x_correction) = img.at<cv::Vec3b>(y,x);
        }
    }

    return output;
}





/******************************
      GIVEN FUNCTIONS
 ******************************/

/**
 * @brief function loads input image, calls processing function, and saves result
 * @param fname path to input image
 */
void run(const std::string &filename) {

    // window names
    std::string win1 = "Original image";
    std::string win2 = "Result";

    // some images
    cv::Mat inputImage, outputImage;


    // load image
    std::cout << "loading image" << std::endl;
    inputImage = cv::imread(filename);
    std::cout << "done" << std::endl;
    
    // check if image can be loaded
    if (!inputImage.data)
        throw std::runtime_error(std::string("ERROR: Cannot read file ") + filename);

    // show input image
    cv::namedWindow(win1.c_str());
    cv::imshow(win1.c_str(), inputImage);
    
    // do something (reasonable!)
    outputImage = doSomethingThatMyTutorIsGonnaLike(inputImage);
    
    // show result
    cv::namedWindow(win2.c_str());
    cv::imshow(win2.c_str(), outputImage);
    
    // save result
    cv::imwrite("result.jpg", outputImage);
    
    // wait a bit
    cv::waitKey(0);
}


}
