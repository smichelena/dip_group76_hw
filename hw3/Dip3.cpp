//============================================================================
// Name    : Dip3.cpp
// Author      : Ronny Haensch, Andreas Ley
// Version     : 3.0
// Copyright   : -
// Description :
//============================================================================

#include "Dip3.h"

#include <math.h>
#include <stdexcept>

namespace dip3 {

const char *const filterModeNames[NUM_FILTER_MODES] = {
    "FM_SPATIAL_CONVOLUTION",
    "FM_FREQUENCY_CONVOLUTION",
    "FM_SEPERABLE_FILTER",
    "FM_INTEGRAL_IMAGE",
};

/**
 * @brief Generates 1D gaussian filter kernel of given size
 * @param kSize Kernel size (used to calculate standard deviation)
 * @returns The generated filter kernel
 */
cv::Mat_<float> createGaussianKernel1D(int kSize) {

  // TO DO !!!
  cv::Mat kernel = cv::Mat_<float>::zeros(1, kSize);
  CV_Assert(kSize % 2 != 0);
  const int kernel_bound = (kSize - 1) / 2;
  const double sigma = ((float)kSize - 1.0f) / 8.0f;
  const double normalizer = 1 / (sigma * sqrt(2 * M_PI));
  for (int i = -kernel_bound; i <= kernel_bound; i++) {
    kernel.at<float>(kernel_bound + i) =
        normalizer * exp(-(i * i) / (2 * sigma * sigma));
  }
  return kernel;
}

/**
 * @brief Generates 2D gaussian filter kernel of given size
 * @param kSize Kernel size (used to calculate standard deviation)
 * @returns The generated filter kernel
 */
cv::Mat_<float> createGaussianKernel2D(int kSize) {

  // TO DO !!!
  cv::Mat kernel = cv::Mat::zeros(kSize, kSize, CV_32FC1);
  // assert kernel size input is valid
  CV_Assert(kSize % 2 != 0);
  // parameters needed for computation
  const int kernel_bound = (kSize - 1) / 2;
  const float sigma = ((float)kSize - 1.) / 8.0f;
  const float normalizer = 1. / (sigma * sigma * 2. * M_PI);
  for (int i = -kernel_bound; i <= kernel_bound; i++) {
    for (int j = -kernel_bound; j <= kernel_bound; j++) {
      float x = i * i + j * j;
      kernel.at<float>(kernel_bound + i, kernel_bound + j) =
          normalizer * exp(-x / (2 * sigma * sigma));
    }
  }

  return kernel;
}

/**
 * @brief Performes a circular shift in (dx,dy) direction
 * @param in Input matrix
 * @param dx Shift in x-direction
 * @param dy Shift in y-direction
 * @returns Circular shifted matrix
 */
cv::Mat_<float> circShift(const cv::Mat_<float> &in, int dx, int dy) {

  // TO DO !!!
  cv::Mat output = in.clone();
  const int max_x = in.cols;
  const int max_y = in.rows;
  for (int y = 0; y < max_y; y++) {
    for (int x = 0; x < in.cols; x++) {
      output.at<float>((max_y + y + dy) % max_y, (max_x + x + dx) % max_x) =
          in.at<float>(y, x);
    }
  }

  return output;
}


/**
 * @brief Performes convolution by multiplication in frequency domain
 * @param in Input image
 * @param kernel Filter kernel
 * @returns Output image
 */
cv::Mat_<float> frequencyConvolution(const cv::Mat_<float> &in,
                                     const cv::Mat_<float> &kernel) {

  // TO DO !!!
  // step 1: construct kernel matrix for convolution
  cv::Mat frequency_kernel = cv::Mat::zeros(in.rows, in.cols, CV_32FC1);

  // create region of interest within output kernel and copy original kernel
  // into it
  cv::Mat ROI_freq_kernel(frequency_kernel,
                          cv::Rect(0, 0, kernel.cols, kernel.rows));
  kernel.copyTo(ROI_freq_kernel);

  // shift kernel appropriately
  frequency_kernel = circShift(frequency_kernel, -(kernel.cols - 1) / 2,
                               -(kernel.cols - 1) / 2);

  // step 2: set up arrays by computing optimal size and so on
  const int optimal_width = cv::getOptimalDFTSize(in.cols);
  const int optimal_height = cv::getOptimalDFTSize(in.rows);
  cv::Mat temp_kernel = cv::Mat::zeros(optimal_height, optimal_width, CV_32FC1);
  cv::Mat temp_input = cv::Mat::zeros(optimal_height, optimal_width, CV_32FC1);

  // do the same ROI trick as before
  cv::Mat ROI_image(temp_input, cv::Rect(0, 0, in.cols, in.rows));
  in.copyTo(ROI_image);
  cv::Mat ROI_kernel(temp_kernel, cv::Rect(0, 0, in.cols, in.rows));
  frequency_kernel.copyTo(ROI_kernel);

  // set up output
  cv::Mat output = cv::Mat::zeros(in.rows, in.cols, CV_32FC1);

  // perform fast fourier transform
  cv::dft(temp_input, temp_input, 0, in.rows);
  cv::dft(temp_kernel, temp_kernel, 0, temp_kernel.rows);

  // convolve
  cv::mulSpectrums(temp_input, temp_kernel, temp_input, 0);

  // inverse transform in frequency domain
  cv::dft(temp_input, temp_input, cv::DFT_INVERSE + cv::DFT_SCALE, output.rows);

  // copy to output
  temp_input(cv::Rect(0, 0, output.cols, output.rows)).copyTo(output);

  return output;
}

/**
 * @brief  Performs UnSharp Masking to enhance fine image structures
 * @param in The input image
 * @param filterMode How convolution for smoothing operation is done
 * @param size Size of used smoothing kernel
 * @param thresh Minimal intensity difference to perform operation
 * @param scale Scaling of edge enhancement
 * @returns Enhanced image
 */
cv::Mat_<float> usm(const cv::Mat_<float> &in, FilterMode filterMode, int size,
                    float thresh, float scale) {
  // TO DO !!!

  // use smoothImage(...) for smoothing
  // save original image
  cv::Mat original = in.clone();

  // smooth image
  cv::Mat smoothed = smoothImage(original, size, filterMode);

  // difference image
  cv::Mat difference;
  cv::subtract(original, smoothed, difference);

  // create enhanced image and output
  cv::Mat enhanced = cv::Mat::zeros(in.rows, in.cols, CV_32FC1);
  // perform enhancenment
  for (int y = 0; y < in.rows; y++) {
    for (int x = 0; x < in.cols; x++) // improved version
    {
      if (abs(difference.at<float>(y, x)) < thresh) {
        enhanced.at<float>(y, x) = original.at<float>(y, x);
      } else if (difference.at<float>(y, x) > 0) {
        enhanced.at<float>(y, x) =
            original.at<float>(y, x) +
            scale * (difference.at<float>(y, x) - thresh);
      } else if (difference.at<float>(y, x) < 0) {
        enhanced.at<float>(y, x) =
            original.at<float>(y, x) +
            scale * (difference.at<float>(y, x) + thresh);
      }
      // truncate if less than zero
      if (enhanced.at<float>(y, x) < 0) {
        enhanced.at<float>(y, x) = 0;
      }
    }
  }

  // apply truncation
  cv::Mat output = enhanced.clone();

  cv::threshold(enhanced, output, 255.0f, 255.0f, cv::THRESH_TRUNC);

  return output; // output;
}

/**
 * @brief Convolution in spatial domain
 * @param src Input image
 * @param kernel Filter kernel
 * @returns Convolution result
 */
cv::Mat_<float> spatialConvolution(const cv::Mat_<float> &src,
                                   const cv::Mat_<float> &kernel) {

  // Hopefully already DONE, copy from last homework

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
 * @brief Convolution in spatial domain by seperable filters
 * @param src Input image
 * @param size Size of filter kernel
 * @returns Convolution result
 */
cv::Mat_<float> separableFilter(const cv::Mat_<float> &src,
                                const cv::Mat_<float> &kernel) {

  // TO DO !!!
  cv::Mat input = src.clone();

  // blur input image horizontally
  input = spatialConvolution(input, kernel);

  // write transposed to temp image
  cv::Mat temp;
  cv::transpose(input, temp);

  // blur temp
  temp = spatialConvolution(temp, kernel);

  // write to putput
  cv::Mat output;
  cv::transpose(temp, output);

  return output;
}

/**
 * @brief Convolution in spatial domain by integral images
 * @param src Input image
 * @param size Size of filter kernel
 * @returns Convolution result
 */
cv::Mat_<float> satFilter(const cv::Mat_<float> &src, int size) {

  // optional
  // step 1: construct kernel matrix for convolution
  cv::Mat integral_image = cv::Mat::zeros(src.rows + 1, src.cols + 1, CV_32FC1);

  // create region of interest within integral image and copy original image
  // into it
  cv::Mat ROI_integral_image(integral_image,
                             cv::Rect(1, 1, src.cols, src.rows));
  src.copyTo(ROI_integral_image);

  for (int x = 1; x < integral_image.cols; x++) {
    for (int y = 1; y < integral_image.rows; y++) {
      integral_image.at<float>(y, x) = integral_image.at<float>(y, x - 1) +
                                       integral_image.at<float>(y - 1, x);
    }
  }

  cv::Mat output = src.clone();
  const float normalizer = static_cast<float>(1 / (size * size));
  const int max_x = src.cols;
  const int max_y = src.rows;
  const int p = (size - 1) / 2;
  for (int x = 1; x < integral_image.cols; x++) {
    for (int y = 1; y < integral_image.rows; y++) {
      auto A = (max_y + y - size) % max_y;
      auto B = (max_x + x - size) % max_x;
      auto C = (max_y + y + size) % max_y;
      auto D = (max_x + x + size) % max_x;
      output.at<float>(y, x) =
          normalizer *
          (integral_image.at<float>(A, B) + integral_image.at<float>(C, D) -
           integral_image.at<float>(A, D) - integral_image.at<float>(C, B));
    }
  }

  return output;
}

/* *****************************
GIVEN FUNCTIONS
***************************** */

/**
 * @brief Performs a smoothing operation but allows the algorithm to be chosen
 * @param in Input image
 * @param size Size of filter kernel
 * @param type How is smoothing performed?
 * @returns Smoothed image
 */
cv::Mat_<float> smoothImage(const cv::Mat_<float> &in, int size,
                            FilterMode filterMode) {
  switch (filterMode) {
  case FM_SPATIAL_CONVOLUTION:
    return spatialConvolution(
        in, createGaussianKernel2D(size)); // 2D spatial convolution
  case FM_FREQUENCY_CONVOLUTION:
    return frequencyConvolution(
        in, createGaussianKernel2D(
                size)); // 2D convolution via multiplication in frequency domain
  case FM_SEPERABLE_FILTER:
    return separableFilter(in,
                           createGaussianKernel1D(size)); // seperable filter
  case FM_INTEGRAL_IMAGE:
    return satFilter(in, size); // integral image
  default:
    throw std::runtime_error("Unhandled filter type!");
  }
}

} // namespace dip3
