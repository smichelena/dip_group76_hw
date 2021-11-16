#include <iostream>
#include <string>
#include <vector>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv2/core/matx.hpp>
#include <cstdint>
#include "Pooling.h"

using namespace std;
using namespace cv;


Mat_<float> spatialConvolution(const Mat_<float>& src, const Mat_<float>& kernel)
{   
    int stride = 1;
    int kernel_size = kernel.rows;

    // create padding
    Mat target = Mat::zeros((kernel_size-1) + src.rows, (kernel_size-1) + src.cols, src.type());

    // ---------- 1
    Rect roi1 = Rect(((kernel_size-1)/2), ((kernel_size-1)/2), src.cols, src.rows);

    Mat dst_roi1;
    dst_roi1 = target(roi1);

    src.copyTo(dst_roi1);


    // create output image
    int w, h;

    w = ceil((src.cols) / stride);
    h = ceil((src.rows) / stride);
    
    Mat out_img(Size(w, h), CV_32FC1);

    float Bpixel_sum;
    Bpixel_sum = 0;

    float Bpoint, Apoint;

    int i, j;

    for(int r = (kernel_size - 1)/2; r < target.rows - (kernel_size - 1)/2; r++) {

        for(int c = (kernel_size - 1)/2; c < target.cols - (kernel_size - 1)/2; c++) {

            if ((r % stride == 0) && (c % stride == 0)){
                i = r-((kernel_size-1)/2);
                j = c-((kernel_size-1)/2);

                float* ptr_res = out_img.ptr<float>(i);

                // iterate through row space
                if (r < target.rows - (kernel_size - 1)/2){

                    for(int shift_row = 0; shift_row < kernel_size; shift_row++){

                        // Set pointer to pixel's row in target mat
                        float* ptr_target = target.ptr<float>((r-((kernel_size-1)/2)) + shift_row);
                        const float* ptr_kernel = kernel.ptr<const float>((kernel_size-1) - shift_row);

                        if (c < target.cols - (kernel_size - 1)/2){

                            for(int shift_column = 0; shift_column < kernel_size; shift_column++){

                                // Set pointer to pixel's row in target mat
                                Bpoint = ptr_target[(c-((kernel_size-1)/2)) + shift_column];
                                Apoint = ptr_kernel[(kernel_size-1) - shift_column];

                                cout << "\n" << " at target position " << (r-((kernel_size-1)/2)) + shift_row << ", " << (c-((kernel_size-1)/2)) + shift_column << " Bpoint is " << Bpoint << endl;

                                Bpixel_sum += (Bpoint * Apoint);

                            }
                        }

                    }

                }
                if (i == 0 || j == 0 || i == (kernel_size*kernel_size)-1 || j == (kernel_size*kernel_size)-1){
                    Bpixel_sum = 0;
                }
                //cout << " at target position " << r << ", " << c << " , into res position " << i << ", " << j << " and pisel sum " << Bpixel_sum << endl;

                ptr_res[j] = Bpixel_sum;
                Bpixel_sum = 0;


            }
            
        }
    }

    return out_img;

}


Mat medianFiltering(Mat image, int kernel_size)
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

                // iterate through row space
                if (r < target.rows - (kernel_size - 1)/2){

                    for(int shift_row = 0; shift_row < kernel_size; shift_row++){

                        // Set pointer to pixel's row in target mat
                        uchar* ptr_target = target.ptr<uchar>((r-((kernel_size-1)/2)) + shift_row);

                        if (c < target.cols - (kernel_size - 1)/2){

                            for(int shift_column = 0; shift_column < kernel_size; shift_column++){

                                // Set pointer to pixel's row in target mat
                                
                                Bpoint = ptr_target[(c-((kernel_size-1)/2)) + shift_column];

                                medi.at((shift_column+1)*(shift_row+1)-1) = Bpoint;

                            }
                        }

                    }

                }

                sort(medi.begin(), medi.end());
                int mid = ceil((kernel_size*kernel_size)/2);

                ptr_res[j] = medi.at(mid);

            }
            
        }
    }

    return out_img;

}

float gaussian(float x, float m, float s)
{
    //static const float inv_sqrt_2pi = 0.3989422804014327;
    float a = (x - m) / s;
    //cout << "a : " << a << " , inner : " << -0.5f * a * a << endl;

    return exp(-0.5f * a * a);
}


Mat_<float> bilateralFilters(const cv::Mat_<float>& src, int kSize, float sigma_spatial, float sigma_radiometric)
{   
    int stride = 1;
    int kernel_size = kSize;

    float hspat, hrad, dis_p, dis_val, cpd_p, cpd_val, weight, pixel_weight;

    // create padding
    Mat target = Mat::zeros((kernel_size-1) + src.rows, (kernel_size-1) + src.cols, src.type());

    // ---------- 1
    Rect roi1 = Rect(((kernel_size-1)/2), ((kernel_size-1)/2), src.cols, src.rows);

    Mat dst_roi1;
    dst_roi1 = target(roi1);

    src.copyTo(dst_roi1);


    // create output image
    int w, h;

    w = ceil((src.cols) / stride);
    h = ceil((src.rows) / stride);
    
    Mat out_img(Size(w, h), CV_32FC1);

    double min, max;
    minMaxLoc(target, &min, &max);

    cout << " min " << min << " , max " << max << endl;

    float Bpoint, center;

    int i, j;

    for(int r = (kernel_size - 1)/2; r < target.rows - (kernel_size - 1)/2; r++) {

        for(int c = (kernel_size - 1)/2; c < target.cols - (kernel_size - 1)/2; c++) {

            if ((r % stride == 0) && (c % stride == 0)){
                i = r-((kernel_size-1)/2);
                j = c-((kernel_size-1)/2);

                float* ptr_res = out_img.ptr<float>(i);

                // iterate through row space
                if (r < target.rows - (kernel_size - 1)/2){

                    for(int shift_row = 0; shift_row < kernel_size; shift_row++){

                        // Set pointer to pixel's row in target mat
                        float* ptr_target = target.ptr<float>((r-((kernel_size-1)/2)) + shift_row);
                        float* ptr_center = target.ptr<float>((r-((kernel_size-1)/2)) + shift_row);

                        if (c < target.cols - (kernel_size - 1)/2){

                            for(int shift_column = 0; shift_column < kernel_size; shift_column++){

                                // Set pointer to pixel's row in target mat
                                Bpoint = ptr_target[(c-((kernel_size-1)/2)) + shift_column];
                                center = ptr_center[c];

                                //cout << "\n" << " at target position " << (r-((kernel_size-1)/2)) + shift_row << ", " << (c-((kernel_size-1)/2)) + shift_column << " Bpoint is " << Bpoint << endl;

                                dis_p = (float) (((kernel_size-1)/2) + shift_column)*(((kernel_size-1)/2) + shift_column);
                                dis_val = (float) (Bpoint - center) * (Bpoint - center);

                                hspat = gaussian(dis_p, 0.0, sigma_spatial);
                                hrad = gaussian(dis_val, 0.0, sigma_radiometric);

                                pixel_weight += hspat * hrad * Bpoint;
                                weight += hspat * hrad;
                                
                                /*cout << "\n" << endl;
                                cout << "dis_p : " << dis_p << endl;
                                cout << "dis_val : " << dis_val << endl;
                                cout << "hspat : " << hspat << endl;
                                cout << "hrad : " << hrad << endl;
                                cout << "pixel_weight  : " << pixel_weight  << endl;
                                cout << "weight  : " << weight  << endl;*/

                            }
                        }

                    }

                }
                /*if (i == 0 || j == 0 || i == (kernel_size*kernel_size)-1 || j == (kernel_size*kernel_size)-1){
                    Bpixel_sum = 0;
                }*/
                //cout << " at target position " << r << ", " << c << " , into res position " << i << ", " << j << " and pisel sum " << Bpixel_sum << endl;

                ptr_res[j] = pixel_weight/weight;
                pixel_weight = 0;
                weight = 0;


            }
            
        }
    }

    return out_img;

}