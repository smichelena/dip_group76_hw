//============================================================================
// Name        : Dip2.cpp
// Author      : Ronny Haensch
// Version     : 2.0
// Copyright   : -
// Description :
//============================================================================

#include "Dip2.h"
#include <math.h>
namespace dip2 {

/**
 * @brief Convolution in spatial domain.
 * @details Performs spatial convolution of image and filter kernel.
 * @param src Input image
 * @param kernel Filter kernel
 * @returns Convolution result
 */
cv::Mat_<float> spatialConvolution(const cv::Mat_<float> &src,
                                   const cv::Mat_<float> &kernel) {
  // TO DO !!

  // obtain initial parameters
  const int max_y = src.rows;
  const int max_x = src.cols;
  const int kernel_bound_i = (kernel.rows - 1) / 2;
  const int kernel_bound_j = (kernel.cols - 1) / 2;
  // validate that kernel is odd-sized matrix or vector
  CV_Assert(kernel.cols % 2 != 0);

  // create output image
  cv::Mat output = src.clone();

  // loop through image
  for (int y = 0; y < max_y; y++) {
    for (int x = 0; x < max_x; x++) {
      // convolve
      float conv_sum = 0.0f;
      for (int i = -kernel_bound_i; i <= kernel_bound_i; i++) {
        for (int j = -kernel_bound_j; j <= kernel_bound_j; j++) {
          conv_sum +=
              src.at<float>((max_y + y + i) % max_y, (max_x + x + j) % max_x) *
              kernel.at<float>(kernel_bound_i - i, kernel_bound_j - j);
        }
      }

      output.at<float>(y, x) = conv_sum;
    }
  }

  return output;
}

/**
 * @brief Moving average filter (aka box filter)
 * @note: you might want to use Dip2::spatialConvolution(...) within this
 * function
 * @param src Input image
 * @param kSize Window size used by local average
 * @returns Filtered image
 */
cv::Mat_<float> averageFilter(const cv::Mat_<float> &src, int kSize) {
  // TO DO !!
  // validate kernel size input
  CV_Assert(kSize % 2 != 0);

  // create kernel matrix
  cv::Mat kernel = cv::Mat::ones(kSize, kSize, CV_32FC1);
  float normalizer = (1 / ((static_cast<float>(kSize * kSize))));
  kernel *= normalizer;

  // return convolution of input with kernel
  return spatialConvolution(src, kernel);
}

/**
 * @brief Median filter
 * @param src Input image
 * @param kSize Window size used by median operation
 * @returns Filtered image
 */
cv::Mat_<float> medianFilter(const cv::Mat_<float> &src, int kSize) {

  // obtain initial parameters
  const int kernel_bound = (kSize - 1) / 2;
  const int max_y = src.rows;
  const int max_x = src.cols;

  // create output image
  cv::Mat output = src.clone();

  // loop through image
  for (int y = 0; y < max_y; y++) {
    for (int x = 0; x < max_x; x++) {

      // filter
      std::vector<float> median_vector = {};
      int counter = 0;
      for (int i = -kernel_bound; i <= kernel_bound; i++) {
        for (int j = -kernel_bound; j <= kernel_bound; j++) {
          median_vector.push_back(
              src.at<float>((max_y + y + i) % max_y, (max_x + x + j) % max_x));
          counter++;
        }
      }

      std::sort(median_vector.begin(), median_vector.end());
      counter = (counter % 2 == 0) ? counter / 2 : (counter - 1) / 2;
      output.at<float>(y, x) = median_vector[counter];
    }
  }

  return output;
}

/**
 * @brief Bilateral filer
 * @param src Input image
 * @param kSize Size of the kernel
 * @param sigma_spatial Standard-deviation of the spatial kernel
 * @param sigma_radiometric Standard-deviation of the radiometric kernel
 * @returns Filtered image
 */
cv::Mat_<float> bilateralFilter(const cv::Mat_<float> &src, int kSize,
                                float sigma_spatial, float sigma_radiometric) {
  // TO DO !!

  // obtain initial parameters
  const int max_y = src.rows;
  const int max_x = src.cols;
  const int kernel_bound = (kSize - 1) / 2;

  // you only ever need the squares
  const float sigma_s = sigma_spatial * sigma_spatial;
  const float sigma_r = sigma_radiometric * sigma_radiometric;

  // validate that kernel is a square, odd-sized matrix
  CV_Assert(kSize % 2 != 0);

  // create output image
  cv::Mat output = cv::Mat::zeros(src.rows, src.cols, CV_32FC1);

  // spatial kernel is independent of x and y
  cv::Mat spatial_kernel = cv::Mat::zeros(kSize, kSize, CV_32FC1);
  for (int i = -kernel_bound; i <= kernel_bound; i++) {
    for (int j = -kernel_bound; j <= kernel_bound; j++) {
      float x = static_cast<float>(i * i + j * j);
      spatial_kernel.at<float>(kernel_bound + i, kernel_bound + j) =
          (1 / (2 * M_PI * sigma_s)) * exp(-x / (2 * sigma_s));
    }
  }

  // loop through image
  for (int y = 0; y < max_y; y++) {
    for (int x = 0; x < max_x; x++) {

      float conv_sum = 0.0f;
      float normalizer_sum = 0.0f;
      float radiometric_kernel;
      // construct kernels
      for (int i = -kernel_bound; i <= kernel_bound; i++) {
        for (int j = -kernel_bound; j <= kernel_bound; j++) {
          float range_dist =
              src.at<float>(y, x) -
              src.at<float>((max_y + y + i) % max_y, (max_x + x + j) % max_x);

          range_dist *= range_dist;
          radiometric_kernel =
              (1 / (2 * M_PI * sigma_r)) * exp(-range_dist / (2 * sigma_r));
          normalizer_sum +=
              radiometric_kernel *
              spatial_kernel.at<float>(kernel_bound + i, kernel_bound + j);
          conv_sum +=
              radiometric_kernel *
              spatial_kernel.at<float>(kernel_bound + i, kernel_bound + j) *
              src.at<float>((max_y + y + i) % max_y, (max_x + x + j) % max_x);
        }
      }

      output.at<float>(y, x) = conv_sum / normalizer_sum;
    }
  }

  return output;
}

/**
 * @brief Non-local means filter
 * @note: This one is optional!
 * @param src Input image
 * @param searchSize Size of search region
 * @param sigma Optional parameter for weighting function
 * @returns Filtered image
 */
cv::Mat_<float> nlmFilter(const cv::Mat_<float> &src, int searchSize,
                          double sigma) {
  auto output = src.clone();
  auto averaged =
      spatialConvolution(src, cv::Mat::ones(searchSize, searchSize, CV_32FC1));
  const auto max_y = src.rows;
  const auto max_x = src.cols;

  for (int y = 0; y < max_y; y++) {
    for (int x = 0; x < max_x; x++) {
      // convolve
      float conv_sum = 0.0f;
      float norm_sum = 0.0f;
      for (int i = -searchSize; i <= searchSize; i++) {
        for (int j = -searchSize; j <= searchSize; j++) {
          auto q_y = (max_y + y + i) % max_y;
          auto q_x = (max_x + x + j) % max_x;
          auto argument = -std::pow(std::abs(averaged.at<float>(q_y, q_x) -
                                             averaged.at<float>(y, x)),
                                    2) /
                          std::pow(sigma, 2);
          conv_sum += src.at<float>(q_y, q_x) * exp(argument);
          norm_sum += exp(argument);
        }
      }

      output.at<float>(y, x) = conv_sum / norm_sum;
    }
  }
  return output;
}

/**
 * @brief Chooses the right algorithm for the given noise type
 * @note: Figure out what kind of noise NOISE_TYPE_1 and NOISE_TYPE_2 are and
 * select the respective "right" algorithms.
 */
NoiseReductionAlgorithm chooseBestAlgorithm(NoiseType noiseType) {
  switch (noiseType) {
  case NOISE_TYPE_1:
    return NR_MEDIAN_FILTER;
    break;

  case NOISE_TYPE_2:
    return NR_MOVING_AVERAGE_FILTER;
    break;
  default:
    return NR_BILATERAL_FILTER;
    break;
  }
}

cv::Mat_<float>
denoiseImage(const cv::Mat_<float> &src, NoiseType noiseType,
             dip2::NoiseReductionAlgorithm noiseReductionAlgorithm) {
  // TO DO !!
  // for each combination find reasonable filter parameters

  switch (noiseReductionAlgorithm) {
  case dip2::NR_MOVING_AVERAGE_FILTER:
    switch (noiseType) {
    case NOISE_TYPE_1:
      return dip2::averageFilter(src, 7);
    case NOISE_TYPE_2:
      return dip2::averageFilter(src, 7);
    default:
      throw std::runtime_error("Unhandled noise type!");
    }
  case dip2::NR_MEDIAN_FILTER:
    switch (noiseType) {
    case NOISE_TYPE_1:
      return dip2::medianFilter(src, 7);
    case NOISE_TYPE_2:
      return dip2::medianFilter(src, 11);
    default:
      throw std::runtime_error("Unhandled noise type!");
    }
  case dip2::NR_BILATERAL_FILTER:
    switch (noiseType) {
    case NOISE_TYPE_1:
      return dip2::bilateralFilter(src, 21, 25.0f, 155.0f);
    case NOISE_TYPE_2:
      return dip2::bilateralFilter(src, 11, 3.0f, 105.0f);
    default:
      throw std::runtime_error("Unhandled noise type!");
    }
    case dip2::NR_NLM_FILTER:
    switch (noiseType) {
    case NOISE_TYPE_1:
      return dip2::nlmFilter(src, 25, 12.0);
    case NOISE_TYPE_2:
      return dip2::nlmFilter(src, 31, 15.0);
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
    "NR_NLM_FILTER",
};

} // namespace dip2
