//============================================================================
// Name        : Dip4.cpp
// Author      : Ronny Haensch, Andreas Ley
// Version     : 3.0
// Copyright   : -
// Description :
//============================================================================

#include "Dip4.h"

namespace dip4 {

using namespace std::complex_literals;

/*

===== std::complex cheat sheet =====

Initialization:

std::complex<float> a(1.0f, 2.0f);
std::complex<float> a = 1.0f + 2.0if;

Common Operations:

std::complex<float> a, b, c;

a = b + c;
a = b - c;
a = b * c;
a = b / c;

std::sin, std::cos, std::tan, std::sqrt, std::pow, std::exp, .... all work as
expected

Access & Specific Operations:

std::complex<float> a = ...;

float real = a.real();
float imag = a.imag();
float phase = std::arg(a);
float magnitude = std::abs(a);
float squared_magnitude = std::norm(a);

std::complex<float> complex_conjugate_a = std::conj(a);

*/

/**
 * @brief Computes the complex valued forward DFT of a real valued input
 * @param input real valued input
 * @return Complex valued output, each pixel storing real and imaginary parts
 */
cv::Mat_<std::complex<float>> DFTReal2Complex(const cv::Mat_<float> &input) {
  // TO DO !!!
  cv::Mat_<std::complex<float>> output;
  cv::dft(input, output, cv::DFT_COMPLEX_OUTPUT);
  return output;
}

/**
 * @brief Computes the real valued inverse DFT of a complex valued input
 * @param input Complex valued input, each pixel storing real and imaginary
 * parts
 * @return Real valued output
 */
cv::Mat_<float> IDFTComplex2Real(const cv::Mat_<std::complex<float>> &input) {
  // TO DO !!!
  cv::Mat_<float> output;
  cv::dft(input, output, cv::DFT_INVERSE + cv::DFT_REAL_OUTPUT + cv::DFT_SCALE);
  return output;
}

/**
 * @brief Performes a circular shift in (dx,dy) direction
 * @param in Input matrix
 * @param dx Shift in x-direction
 * @param dy Shift in y-direction
 * @return Circular shifted matrix
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
 * @brief Applies a filter (in frequency domain)
 * @param input Image in frequency domain (complex valued)
 * @param filter Filter in frequency domain (complex valued), same size as input
 * @return The filtered image, complex valued, in frequency domain
 */
cv::Mat_<std::complex<float>>
applyFilter(const cv::Mat_<std::complex<float>> &input,
            const cv::Mat_<std::complex<float>> &filter) {
  // TO DO !!
  cv::Mat in = input.clone();
  cv::Mat fil = filter.clone();
  cv::mulSpectrums(in, fil, in, 0);
  return in;
}

/**
 * @brief Computes the thresholded inverse filter
 * @param input Blur filter in frequency domain (complex valued)
 * @param eps Factor to compute the threshold (relative to the max amplitude)
 * @return The inverse filter in frequency domain (complex valued)
 */
cv::Mat_<std::complex<float>>
computeInverseFilter(const cv::Mat_<std::complex<float>> &input,
                     const float eps) {
  // TO DO !!!
  // build magnitude matrix
  cv::Mat magnitude = cv::Mat::zeros(input.rows, input.cols, CV_32FC1);
  for (int y = 0; y < input.rows; y++) {
    for (int x = 0; x < input.cols; x++) {
      magnitude.at<float>(y, x) = std::abs(input.at<std::complex<float>>(y, x));
    }
  }

  // find max of magnitude matrix and define threshold
  double minVal;
  double maxVal;
  cv::Point minLoc;
  cv::Point maxLoc;
  cv::minMaxLoc(magnitude, &minVal, &maxVal, &minLoc, &maxLoc);
  float thresh = eps * maxVal;

  // build output
  cv::Mat output = input.clone();
  for (int y = 0; y < input.rows; y++) {
    for (int x = 0; x < input.cols; x++) {
      output.at<std::complex<float>>(y, x) =
          (magnitude.at<float>(y, x) >= thresh)
              ? std::pow(input.at<std::complex<float>>(y, x), -1)
              : 1 / thresh;
    }
  }

  return output;
}

/**
 * @brief Function applies the inverse filter to restorate a degraded image
 * @param degraded Degraded input image
 * @param filter Filter which caused degradation
 * @param eps Factor to compute the threshold (relative to the max amplitude)
 * @return Restorated output image
 */
cv::Mat_<float> inverseFilter(const cv::Mat_<float> &degraded,
                              const cv::Mat_<float> &filter, const float eps) {
  // TO DO !!!
  // step 1: construct kernel matrix for convolution
  cv::Mat kernel = cv::Mat::zeros(degraded.rows, degraded.cols, CV_32FC1);

  // create region of interest within output kernel and copy original kernel
  // into it
  cv::Mat ROI_freq_kernel(kernel, cv::Rect(0, 0, filter.cols, filter.rows));
  filter.copyTo(ROI_freq_kernel);

  // shift kernel appropriately
  kernel = circShift(kernel, -(filter.cols - 1) / 2, -(filter.cols - 1) / 2);

  // get complex spectrae
  cv::Mat_<std::complex<float>> frequency_kernel = DFTReal2Complex(kernel);
  cv::Mat_<std::complex<float>> frequency_degraded = DFTReal2Complex(degraded);

  // compute inverse filter
  cv::Mat_<std::complex<float>> inverse_filter =
      computeInverseFilter(frequency_kernel, eps);

  // apply filter
  cv::Mat_<std::complex<float>> frequency_output =
      applyFilter(frequency_degraded, inverse_filter);

  // apply inverse dft
  cv::Mat_<float> output = IDFTComplex2Real(frequency_output);

  return output;
}

/**
 * @brief Computes the Wiener filter
 * @param input Blur filter in frequency domain (complex valued)
 * @param snr Signal to noise ratio
 * @return The wiener filter in frequency domain (complex valued)
 */
cv::Mat_<std::complex<float>>
computeWienerFilter(const cv::Mat_<std::complex<float>> &input,
                    const float snr) {
  // TO DO !!!
  // build output
  auto k = (1 / std::pow(snr, 2));
  cv::Mat output = input.clone();
  for (int y = 0; y < input.rows; y++) {
    for (int x = 0; x < input.cols; x++) {
      std::complex<float> nenner(
          std::pow(std::abs(input.at<std::complex<float>>(y, x)), 2) + k, 0);
      output.at<std::complex<float>>(y, x) =
          (std::conj(input.at<std::complex<float>>(y, x))) / nenner;
    }
  }
  return output;
}

/**
 * @brief Function applies the wiener filter to restore a degraded image
 * @param degraded Degraded input image
 * @param filter Filter which caused degradation
 * @param snr Signal to noise ratio of the input image
 * @return Restored output image
 */
cv::Mat_<float> wienerFilter(const cv::Mat_<float> &degraded,
                             const cv::Mat_<float> &filter, float snr) {
  // TO DO !!!
  // step 1: construct kernel matrix for convolution
  cv::Mat kernel = cv::Mat::zeros(degraded.rows, degraded.cols, CV_32FC1);

  // create region of interest within output kernel and copy original kernel
  // into it
  cv::Mat ROI_freq_kernel(kernel, cv::Rect(0, 0, filter.cols, filter.rows));
  filter.copyTo(ROI_freq_kernel);

  // shift kernel appropriately
  kernel = circShift(kernel, -(filter.cols - 1) / 2, -(filter.cols - 1) / 2);

  // get complex spectrae
  cv::Mat_<std::complex<float>> frequency_kernel = DFTReal2Complex(kernel);
  cv::Mat_<std::complex<float>> frequency_degraded = DFTReal2Complex(degraded);

  // compute inverse filter
  cv::Mat_<std::complex<float>> wiener_filter =
      computeWienerFilter(frequency_kernel, snr);

  // apply filter
  cv::Mat_<std::complex<float>> frequency_output =
      applyFilter(frequency_degraded, wiener_filter);

  // apply inverse dft
  cv::Mat_<float> output = IDFTComplex2Real(frequency_output);

  return output;
}

/* *****************************
  GIVEN FUNCTIONS
***************************** */

/**
 * function degrades the given image with gaussian blur and additive gaussian
 * noise
 * @param img Input image
 * @param degradedImg Degraded output image
 * @param filterDev Standard deviation of kernel for gaussian blur
 * @param snr Signal to noise ratio for additive gaussian noise
 * @return The used gaussian kernel
 */
cv::Mat_<float> degradeImage(const cv::Mat_<float> &img,
                             cv::Mat_<float> &degradedImg, float filterDev,
                             float snr) {

  int kSize = round(filterDev * 3) * 2 - 1;

  cv::Mat gaussKernel = cv::getGaussianKernel(kSize, filterDev, CV_32FC1);
  gaussKernel = gaussKernel * gaussKernel.t();

  cv::Mat imgs = img.clone();
  cv::dft(imgs, imgs, img.rows);
  cv::Mat kernels = cv::Mat::zeros(img.rows, img.cols, CV_32FC1);
  int dx, dy;
  dx = dy = (kSize - 1) / 2.;
  for (int i = 0; i < kSize; i++)
    for (int j = 0; j < kSize; j++)
      kernels.at<float>((i - dy + img.rows) % img.rows,
                        (j - dx + img.cols) % img.cols) =
          gaussKernel.at<float>(i, j);
  cv::dft(kernels, kernels);
  cv::mulSpectrums(imgs, kernels, imgs, 0);
  cv::dft(imgs, degradedImg, cv::DFT_INVERSE + cv::DFT_SCALE, img.rows);

  cv::Mat mean, stddev;
  cv::meanStdDev(img, mean, stddev);

  cv::Mat noise = cv::Mat::zeros(img.rows, img.cols, CV_32FC1);
  cv::randn(noise, 0, stddev.at<double>(0) / snr);
  degradedImg = degradedImg + noise;
  cv::threshold(degradedImg, degradedImg, 255, 255, cv::THRESH_TRUNC);
  cv::threshold(degradedImg, degradedImg, 0, 0, cv::THRESH_TOZERO);

  return gaussKernel;
}

} // namespace dip4
