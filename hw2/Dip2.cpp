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

    float local_average(const cv::Mat_<float> &src, int x_begin, int x_end, int y_begin, int y_end)
    {
        //compute local window average
        float local_avg = 0.0f;
        int counter = 0;
        for (int k = y_begin; k < y_end; k++)
        {
            for (int t = x_begin; t < x_end; t++)
            {
                local_avg += src.at<float>(k, t);
                counter++;
            }
        }
        return local_avg / counter;
    }

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
        const int max_x = src.cols;

        //validate that kernel is a square, odd-sized matrix
        CV_Assert(w == kernel.rows && w % 2 != 0);

        //create output image
        cv::Mat output = src;

        //loop through image
        for (int y = 0; y < max_y; y++)
        {
            for (int x = 0; x < max_x; x++)
            {

                //edge case handling
                int x_begin, y_begin, x_end, y_end;
                float local_avg = 0.0f;
                //if we encounter an edge case we use the local avg.
                //upper left corner
                if (y - (w - 1) / 2 < 0 && x - (w - 1) / 2 < 0)
                {
                    x_begin = 0;
                    y_begin = 0;
                    x_end = x + (w - 1) / 2;
                    y_end = y + (w - 1) / 2;
                    local_avg = local_average(src, x_begin, x_end, y_begin, y_end);
                }
                //upper right corner
                else if (y - (w - 1) / 2 < 0 && x + (w - 1) / 2 > max_x - 1)
                {
                    x_begin = x - (w - 1) / 2;
                    x_end = max_x;
                    y_begin = 0;
                    y_end = y + (w - 1) / 2;
                    local_avg = local_average(src, x_begin, x_end, y_begin, y_end);
                }
                //lower right corner
                else if (y + (w - 1) / 2 > max_y - 1 && x + (w - 1) / 2 > max_x - 1)
                {
                    x_begin = x - (w - 1) / 2;
                    x_end = max_x;
                    y_begin = y - (w - 2) / 2;
                    y_end = max_y;
                    local_avg = local_average(src, x_begin, x_end, y_begin, y_end);
                }
                //lower left corner
                else if (y + (w - 1) / 2 > max_y - 1 && x - (w - 1) / 2 < 0)
                {
                    x_begin = 0;
                    x_end = x + (w - 1) / 2;
                    y_begin = y - (w - 1) / 2;
                    y_end = max_y;
                    local_avg = local_average(src, x_begin, x_end, y_begin, y_end);
                }
                //lower edge
                else if (y + (w - 1) / 2 > max_y - 1 && x + (w - 1) / 2 < max_x && x - (w - 1) / 2 >= 0)
                {
                    y_begin = y - (w - 1) / 2;
                    y_end = max_y;
                    x_begin = x - (w - 1) / 2;
                    x_end = x + (w - 1) / 2;
                    local_avg = local_average(src, x_begin, x_end, y_begin, y_end);
                }
                //upper edge
                else if (y - (w - 1) / 2 < 0 && x + (w - 1) / 2 < max_x && x - (w - 1) / 2 >= 0)
                {
                    y_begin = 0;
                    y_end = y + (w - 1) / 2;
                    x_begin = x - (w - 1) / 2;
                    x_end = x + (w - 1) / 2;
                    local_avg = local_average(src, x_begin, x_end, y_begin, y_end);
                }
                //left edge
                else if (x - (w - 1) / 2 < 0 && y + (w - 1) / 2 < max_y && y - (w - 1) / 2 >= 0)
                {
                    x_begin = 0;
                    x_end = x + (w - 1) / 2;
                    y_begin = y - (w - 1) / 2;
                    y_end = y + (w - 1) / 2;
                    local_avg = local_average(src, x_begin, x_end, y_begin, y_end);
                }
                //right edge
                else if (x + (w - 1) / 2 > max_x - 1 && y + (w - 1) / 2 < max_y && y - (w - 1) / 2 >= 0)
                {
                    x_begin = x - (w - 1) / 2;
                    x_end = max_x;
                    y_begin = y - (w - 1) / 2;
                    y_end = y + (w - 1) / 2;
                    local_avg = local_average(src, x_begin, x_end, y_begin, y_end);
                }

                //convolve
                float conv_sum = 0.0f;
                for (int i = -(w - 1) / 2; i <= (w - 1) / 2; i++)
                {
                    for (int j = -(w - 1) / 2; j <= (w - 1) / 2; j++)
                    {
                        if (y + i < 0 || x + j < 0 || y + i > max_y - 1 || x + j > max_x - 1)
                        {
                            conv_sum += local_avg * kernel.at<float>((w - 1) / 2 - i, (w - 1) / 2 - j);
                        }
                        else
                        {
                            conv_sum += (src.at<float>(y + i, x + j)) * kernel.at<float>((w - 1) / 2 - i, (w - 1) / 2 - j);
                        }
                    }
                }
                output.at<float>(y, x) = conv_sum;
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
        //validate kernel size input
        CV_Assert(kSize % 2 != 0);

        //create kernel matrix
        cv::Mat kernel = cv::Mat::ones(kSize, kSize, CV_32FC1);
        float normalizer = (1 / ((float)kSize * kSize));
        kernel *= normalizer;

        //return convolution of input with kernel
        return spatialConvolution(src, kernel);
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
        return NR_MOVING_AVERAGE_FILTER; //pls ignore if youre writing this part, I put this here just for testing
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
                return dip2::averageFilter(src, 7); //please ignore paramenters here, I put that in just for testing
            case NOISE_TYPE_2:
                return dip2::averageFilter(src, 9);
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
