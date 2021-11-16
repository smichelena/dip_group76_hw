#include <iostream>
#include <string>
#include <vector>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv2/core/matx.hpp>
#include <cstdint>
#include "Pooling.h"

using namespace std;
using namespace cv;

Mat spatialConvolution(Mat image, Mat kernel, int weight)
{   
    int stride = 1;
    int kernel_size = kernel.rows;

    // create padding
    Mat fill1 = Mat::ones(((kernel_size-1)/2), image.cols, CV_8UC1);
    Mat v_fill = fill1 * 0;

    Mat pre_target = Mat(image.rows, image.cols, CV_8UC1);

    vconcat(image, v_fill, pre_target);
    vconcat(v_fill, pre_target, pre_target);

    Mat fill2 = Mat::ones(pre_target.rows, ((kernel_size-1)/2), CV_8UC1);
    Mat h_fill = fill2 * 0;

    Mat target = Mat(pre_target.rows, image.cols, CV_8UC1);
    
    hconcat(pre_target, h_fill, target);
    hconcat(h_fill, target, target);

    //cout << "given target matrix " << target.cols << endl;

    // create output image
    int w, h;

    w = ceil((image.cols) / stride);
    h = ceil((image.rows) / stride);
    
    Mat out_img(Size(w, h), CV_8UC1);

    float Bpixel_sum;
    Bpixel_sum = 0;

    float Bpoint, Apoint;

    int i, j;

    for(int r = (kernel_size - 1)/2; r < target.rows - (kernel_size - 1)/2; r++) {

        for(int c = (kernel_size - 1)/2; c < target.cols - (kernel_size - 1)/2; c++) {

            if ((r % stride == 0) && (c % stride == 0)){
                i = r-((kernel_size-1)/2);
                j = c-((kernel_size-1)/2);

                uchar* ptr_res = out_img.ptr<uchar>(i);

                // iterate through row space
                if (r < target.rows - (kernel_size - 1)/2){

                    for(int shift_row = 0; shift_row < kernel_size; shift_row++){

                        // Set pointer to pixel's row in target mat
                        uchar* ptr_target = target.ptr<uchar>((r-((kernel_size-1)/2)) + shift_row);
                        uchar* ptr_kernel = kernel.ptr<uchar>((kernel_size-1) - shift_row);

                        if (c < target.cols - (kernel_size - 1)/2){

                            for(int shift_column = 0; shift_column < kernel_size; shift_column++){

                                // Set pointer to pixel's row in target mat
                                Bpoint = ptr_target[(c-((kernel_size-1)/2)) + shift_column];
                                Apoint = ptr_kernel[(kernel_size-1) - shift_column];

                                /*cout << "\n" << " at target position " << (r-((kernel_size-1)/2)) + shift_row << ", " << (c-((kernel_size-1)/2)) + shift_column << endl;
                                cout << "Bpoint is " << Bpoint << endl;
                                cout << "Apoint is " << Apoint << endl;*/

                                Bpixel_sum += (float)((Bpoint * Apoint) / (weight));

                            }
                        }

                    }

                }

                /*cout << " at target position " << r << ", " << c << endl;
                cout << " into res position " << i << ", " << j << endl;*/

                uchar Bpixel = (uchar) Bpixel_sum;
                ptr_res[j] = Bpixel;
                Bpixel_sum = 0;

            }
            
        }
    }

    return out_img;

}


Mat medianFilter(Mat image, int kernel_size)
{   
    int stride = 1;

    // create padding
    Mat fill1 = Mat::ones(((kernel_size-1)/2), image.cols, CV_8UC1);
    Mat v_fill = fill1 * 0;

    Mat pre_target = Mat(image.rows, image.cols, CV_8UC1);

    vconcat(image, v_fill, pre_target);
    vconcat(v_fill, pre_target, pre_target);

    Mat fill2 = Mat::ones(pre_target.rows, ((kernel_size-1)/2), CV_8UC1);
    Mat h_fill = fill2 * 0;

    Mat target = Mat(pre_target.rows, image.cols, CV_8UC1);
    
    hconcat(pre_target, h_fill, target);
    hconcat(h_fill, target, target);

    //cout << "given target matrix " << target.cols << endl;

    // create output image
    int w, h;

    w = ceil((image.cols) / stride);
    h = ceil((image.rows) / stride);
    
    Mat out_img(Size(w, h), CV_8UC1);

    float Bpixel_sum;
    Bpixel_sum = 0;

    float Bpoint;

    std::vector<int> medi (kernel_size*kernel_size);

    int i, j;

    for(int r = (kernel_size - 1)/2; r < target.rows - (kernel_size - 1)/2; r++) {

        for(int c = (kernel_size - 1)/2; c < target.cols - (kernel_size - 1)/2; c++) {

            if ((r % stride == 0) && (c % stride == 0)){
                i = r-((kernel_size-1)/2);
                j = c-((kernel_size-1)/2);

                uchar* ptr_res = out_img.ptr<uchar>(i);
                
                /*uchar values[kernel_size*kernel_size];
                uchar * val_ptr = values;*/

                // iterate through row space
                if (r < target.rows - (kernel_size - 1)/2){

                    for(int shift_row = 0; shift_row < kernel_size; shift_row++){

                        // Set pointer to pixel's row in target mat
                        uchar* ptr_target = target.ptr<uchar>((r-((kernel_size-1)/2)) + shift_row);

                        if (c < target.cols - (kernel_size - 1)/2){

                            for(int shift_column = 0; shift_column < kernel_size; shift_column++){

                                // Set pointer to pixel's row in target mat
                                
                                Bpoint = ptr_target[(c-((kernel_size-1)/2)) + shift_column];

                                /*val_ptr = &values[(shift_column+1)*(shift_row+1)-1];  
                                *val_ptr = Bpoint;*/

                                medi.at((shift_column+1)*(shift_row+1)-1) = Bpoint;

                                /*cout << "\n" << "medi position " << (shift_column+1)*(shift_row+1)-1 << endl;

                                cout << "\n" << " at target position " << (r-((kernel_size-1)/2)) + shift_row << ", " << (c-((kernel_size-1)/2)) + shift_column << endl;
                                cout << "Bpoint is " << Bpoint << endl;
                                cout << "Apoint is " << Apoint << endl;*/

                            }
                        }

                    }

                }

                /*cout << " at target position " << r << ", " << c << endl;
                cout << " into res position " << i << ", " << j << endl;*/

                /*uchar n = sizeof(values) / sizeof(values[0]);
                sort(values, values + n);
                uchar x = values[ceil(kernel_size/2)];*/

                sort(medi.begin(), medi.end());
                int mid = ceil((kernel_size*kernel_size)/2);

                /*cout << "\n postion operation " << ceil((kernel_size*kernel_size)/2) << " and " << (kernel_size*kernel_size)/2 << " and " << kernel_size << endl;

                std::cout << "myvector contains:";
                for (unsigned i=0; i<medi.size(); i++)
                std::cout << ' ' << medi.at(i);
                std::cout << '\n';

                cout << "median is chosen for alg " << medi.at(mid) << " for position " << mid << endl;
                cout << "___________________________________________" << endl;*/

                ptr_res[j] = medi.at(mid);

            }
            
        }
    }

    return out_img;

}