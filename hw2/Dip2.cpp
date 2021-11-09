//============================================================================
// Name        : Dip2.cpp
// Author      : Ronny Haensch
// Version     : 2.0
// Copyright   : -
// Description :
//============================================================================

#include "Dip2.h"

namespace dip2
{

    /**
 * @brief Convolution in spatial domain.
 * @details Performs spatial convolution of image and filter kernel.
 * @param src Input image
 * @param kernel Filter kernel
 * @returns Convolution result
 */
    cv::Mat_<float> spatialConvolution(const cv::Mat_<float> &src, const cv::Mat_<float> &kernel)
    {
        // TO DO !!

        //obtain initial parameters
        const int w = kernel.cols;
        const int max_y = src.rows;
        const int max_x = src.rows;
        const int channels = src.channels();

        //create output image
        cv::Mat output = src;

        //loop through channels
        for (int c = 0; c < channels; c++)
        {
            //loop through image
            for (int y = 0; y < max_y; y++)
            {
                for (int x = 0; x < max_x; x++)
                {
                    //local average variable for edge case handling
                    float local_avg = 0.0f;

                    //edge case, outside lower bounds:
                    if (y - (w - 1) / 2 < 0 || x - (w - 1) / 2 < 0)
                    {
                        //compute local window average
                        int x_excess = (w - 1) / 2 - x + 1;
                        int y_excess = (w - 1) / 2 - y + 1;
                        local_avg = 0.0f;
                        for (int k = 0; k < w - x_excess; k++)
                        {
                            for (int t = 0; k < w - y_excess; t++)
                            {
                                local_avg += (float)src.at<cv::Vec3b>(k, t)[c];
                            }
                        }
                        local_avg /= (w - x_excess) * (w - y_excess);
                    }
                    //edge case, outside upper bounds
                    else if (y + (w - 1) / 2 > max_y - 1 || x + (w - 1) / 2 > max_x - 1)
                    {
                        //compute local window average
                        int x_excess = x + (w - 1) / 2 - max_x - 1;
                        int y_excess = y + (w - 1) / 2 - max_y - 1;
                        local_avg = 0.0f;
                        for (int k = x - (w - 1) / 2; k < max_x; k++)
                        {
                            for (int t = y - (w - 1) / 2; k < max_y; t++)
                            {
                                local_avg += (float)src.at<cv::Vec3b>(k, t)[c];
                            }
                        }
                        local_avg /= (w - x_excess) * (w - y_excess);
                    }

                    //convolve
                    float conv_sum = 0.0f;
                    for (int i = -(w - 1) / 2; i <= (w - 1) / 2; i++)
                    {
                        for (int j = -(w - 1) / 2; j <= (w - 1) / 2; j++)
                        {
                            if (y + i < 0 || x - j < 0)
                            {
                                conv_sum += local_avg * kernel.at<float>((w - 1) / 2 - i, (w - 1) / 2 - j);
                            }
                            else if (y + i > max_y || x + j > max_x)
                            {
                                conv_sum += local_avg * kernel.at<float>((w - 1) / 2 - i, (w - 1) / 2 - j);
                            }
                            conv_sum += ((float)src.at<cv::Vec3b>(y + i, x + j)[c]) * kernel.at<float>((w - 1) / 2 - i, (w - 1) / 2 - j);
                        }
                    }
                    output.at<cv::Vec3b>(y, x)[c] = (int)conv_sum; //cast to int ?
                }
            }
        }
        return output;
    }

    /**
 * @brief Moving average filter (aka box filter)
 * @note: you might want to use Dip2::spatialConvolution(...) within this function
 * @param src Input image
 * @param kSize Window size used by local average
 * @returns Filtered image
 */
cv::Mat_<float> averageFilter(const cv::Mat_<float> &src, int kSize)
{
    // TO DO !!
    return src.clone();
}

    /**
 * @brief Median filter
 * @param src Input image
 * @param kSize Window size used by median operation
 * @returns Filtered image
 */
    cv::Mat_<float> medianFilter(const cv::Mat_<float> &src, int kSize)
    {
        // TO DO !!
        return src.clone();
    }

    /**
 * @brief Bilateral filer
 * @param src Input image
 * @param kSize Size of the kernel
 * @param sigma_spatial Standard-deviation of the spatial kernel
 * @param sigma_radiometric Standard-deviation of the radiometric kernel
 * @returns Filtered image
 */
    cv::Mat_<float> bilateralFilter(const cv::Mat_<float> &src, int kSize, float sigma_spatial, float sigma_radiometric)
    {
        // TO DO !!
        return src.clone();
    }

    /**
 * @brief Non-local means filter
 * @note: This one is optional!
 * @param src Input image
 * @param searchSize Size of search region
 * @param sigma Optional parameter for weighting function
 * @returns Filtered image
 */
    cv::Mat_<float> nlmFilter(const cv::Mat_<float> &src, int searchSize, double sigma)
    {
        return src.clone();
    }

    /**
 * @brief Chooses the right algorithm for the given noise type
 * @note: Figure out what kind of noise NOISE_TYPE_1 and NOISE_TYPE_2 are and select the respective "right" algorithms.
 */
    NoiseReductionAlgorithm chooseBestAlgorithm(NoiseType noiseType)
    {
        // TO DO !!
        return (NoiseReductionAlgorithm)-1;
    }

    cv::Mat_<float> denoiseImage(const cv::Mat_<float> &src, NoiseType noiseType, dip2::NoiseReductionAlgorithm noiseReductionAlgorithm)
    {
        // TO DO !!

        // for each combination find reasonable filter parameters

        switch (noiseReductionAlgorithm)
        {
        case dip2::NR_MOVING_AVERAGE_FILTER:
            switch (noiseType)
            {
            case NOISE_TYPE_1:
                return dip2::averageFilter(src, 1);
            case NOISE_TYPE_2:
                return dip2::averageFilter(src, 1);
            default:
                throw std::runtime_error("Unhandled noise type!");
            }
        case dip2::NR_MEDIAN_FILTER:
            switch (noiseType)
            {
            case NOISE_TYPE_1:
                return dip2::medianFilter(src, 1);
            case NOISE_TYPE_2:
                return dip2::medianFilter(src, 1);
            default:
                throw std::runtime_error("Unhandled noise type!");
            }
        case dip2::NR_BILATERAL_FILTER:
            switch (noiseType)
            {
            case NOISE_TYPE_1:
                return dip2::bilateralFilter(src, 1, 1.0f, 1.0f);
            case NOISE_TYPE_2:
                return dip2::bilateralFilter(src, 1, 1.0f, 1.0f);
            default:
                throw std::runtime_error("Unhandled noise type!");
            }
        default:
            throw std::runtime_error("Unhandled filter type!");
        }
    }

    // Helpers, don't mind these

    const char *noiseTypeNames[NUM_NOISE_TYPES] = {
        "NOISE_TYPE_1",
        "NOISE_TYPE_2",
    };

    const char *noiseReductionAlgorithmNames[NUM_FILTERS] = {
        "NR_MOVING_AVERAGE_FILTER",
        "NR_MEDIAN_FILTER",
        "NR_BILATERAL_FILTER",
    };
    
}
