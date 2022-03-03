//============================================================================
// Name        : Dip5.cpp
// Author      : Ronny Haensch, Andreas Ley
// Version     : 3.0
// Copyright   : -
// Description :
//============================================================================

#include "Dip5.h"
#include <numeric>

// for debugging
void print_matrix(const cv::Mat_<float> &in) {
  std::cout << "================================================" << std::endl;
  for (int i = 0; i < in.rows; i++) {
    for (int j = 0; j < in.cols; j++) {
      std::cout << " " << in.at<float>(i, j) << " ";
    }
    std::cout << " " << std::endl;
  }
  std::cout << "================================================" << std::endl;
}

namespace dip5 {

/**
 * @brief Generates gaussian filter kernel of given size
 * @param kSize Kernel size (used to calculate standard deviation)
 * @returns The generated filter kernel
 */
cv::Mat_<float> createGaussianKernel1D(float sigma) {
  unsigned kSize = getOddKernelSizeForSigma(sigma);
  // Hopefully already DONE, copy from last homework, just make sure you compute
  // the kernel size from the given sigma (and not the other way around)
  cv::Mat_<float> kernel = cv::Mat::zeros(1, kSize, CV_32FC1);
  int kernel_bound = (kSize - 1) / 2;
  float normalizer = 1 / (sigma * sqrt(2 * M_PI));
  std::transform(kernel.begin(), kernel.end(), kernel.begin(),
                 [&kernel_bound, &sigma, &normalizer, i = 0](float t) mutable {
                   auto x = i - kernel_bound;
                   auto temp = normalizer *
                               exp(-((float)(x * x)) / (2.0F * sigma * sigma));
                   i++;
                   return temp;
                 });
  return kernel;
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
                                const cv::Mat_<float> &kernelX,
                                const cv::Mat_<float> &kernelY) {
  // Hopefully already DONE, copy from last homework
  // But do mind that this one gets two different kernels for horizontal and
  // vertical convolutions.
  cv::Mat input = src.clone();

  // blur input image horizontally
  input = spatialConvolution(input, kernelX);

  // write transposed to temp image
  cv::Mat temp;
  cv::transpose(input, temp);

  // blur temp
  temp = spatialConvolution(temp, kernelY);

  // write to putput
  cv::Mat output;
  cv::transpose(temp, output);

  return output;
}

/**
 * @brief Creates kernel representing fst derivative of a Gaussian kernel
 * (1-dimensional)
 * @param sigma standard deviation of the Gaussian kernel
 * @returns the calculated kernel
 */
cv::Mat_<float> createFstDevKernel1D(float sigma) {
  unsigned kSize = getOddKernelSizeForSigma(sigma);
  // TO DO !!!
  cv::Mat_<float> original = createGaussianKernel1D(sigma);
  const int kernel_bound = (kSize - 1) / 2;
  // create position array
  cv::Mat_<float> result(1, kSize, CV_32FC1);
  // compute result
  std::transform(original.begin(), original.end(), result.begin(),
                 [&kernel_bound, &sigma, i = 0](float gaussianSample) mutable {
                   auto temp =
                       -(i - kernel_bound) / (sigma * sigma) * gaussianSample;
                   i++;
                   return temp;
                 });
  return result;
}

/**
 * @brief Calculates the directional gradients through convolution
 * @param img The input image
 * @param sigmaGrad The standard deviation of the Gaussian kernel for the
 * directional gradients
 * @param gradX Matrix through which to return the x component of the
 * directional gradients
 * @param gradY Matrix through which to return the y component of the
 * directional gradients
 */
void calculateDirectionalGradients(const cv::Mat_<float> &img, float sigmaGrad,
                                   cv::Mat_<float> &gradX,
                                   cv::Mat_<float> &gradY) {
  // TO DO !!!

  // create gaussian kernel and first derivative
  cv::Mat gaussianKernel = createGaussianKernel1D(sigmaGrad);
  cv::Mat devGaussianKernel = createFstDevKernel1D(sigmaGrad);

  // compute directional gradients
  gradX = separableFilter(img, devGaussianKernel, gaussianKernel);
  gradY = separableFilter(img, gaussianKernel, devGaussianKernel);
}

/**
 * @brief Calculates the structure tensors (per pixel)
 * @param gradX The x component of the directional gradients
 * @param gradY The y component of the directional gradients
 * @param sigmaNeighborhood The standard deviation of the Gaussian kernel for
 * computing the "neighborhood summation".
 * @param A00 Matrix through which to return the A_{0,0} elements of the
 * structure tensor of each pixel.
 * @param A01 Matrix through which to return the A_{0,1} elements of the
 * structure tensor of each pixel.
 * @param A11 Matrix through which to return the A_{1,1} elements of the
 * structure tensor of each pixel.
 */
void calculateStructureTensor(const cv::Mat_<float> &gradX,
                              const cv::Mat_<float> &gradY,
                              float sigmaNeighborhood, cv::Mat_<float> &A00,
                              cv::Mat_<float> &A01, cv::Mat_<float> &A11) {
  // TO DO !!!
  cv::Mat_<float> fx_squared = gradX.clone();
  std::transform(gradX.begin(), gradX.end(), fx_squared.begin(),
                 [](float fx) { return fx * fx; });

  cv::Mat_<float> fy_squared = gradY.clone();
  std::transform(gradY.begin(), gradY.end(), fy_squared.begin(),
                 [](float fy) { return fy * fy; });

  cv::Mat_<float> fxfy = gradX.clone();
  std::transform(gradY.begin(), gradY.end(), gradX.begin(), fxfy.begin(),
                 [](float fx, float fy) { return fx * fy; });

  cv::Mat artifical_kernel = createGaussianKernel1D(sigmaNeighborhood);

  A00 = separableFilter(fx_squared, artifical_kernel, artifical_kernel);
  A01 = separableFilter(fxfy, artifical_kernel, artifical_kernel);
  A11 = separableFilter(fy_squared, artifical_kernel, artifical_kernel);
}

/**
 * @brief Calculates the feature point weight and isotropy from the structure
 * tensors.
 * @param A00 The A_{0,0} elements of the structure tensor of each pixel.
 * @param A01 The A_{0,1} elements of the structure tensor of each pixel.
 * @param A11 The A_{1,1} elements of the structure tensor of each pixel.
 * @param weight Matrix through which to return the weights of each pixel.
 * @param isotropy Matrix through which to return the isotropy of each pixel.
 */
void calculateFoerstnerWeightIsotropy(const cv::Mat_<float> &A00,
                                      const cv::Mat_<float> &A01,
                                      const cv::Mat_<float> &A11,
                                      cv::Mat_<float> &weight,
                                      cv::Mat_<float> &isotropy) {
  cv::Mat_<float> weightTemp = cv::Mat::zeros(A00.size(), CV_32FC1);
  cv::Mat_<float> isotropyTemp = cv::Mat::zeros(A00.size(), CV_32FC1);
  for (int x = 0; x < A00.cols; x++) {
    for (int y = 0; y < A00.rows; y++) {
      float det = A00.at<float>(y, x) * A11.at<float>(y, x) -
                  A01.at<float>(y, x) * A01.at<float>(y, x);
      float trace = A00.at<float>(y, x) + A11.at<float>(y, x);
      weightTemp.at<float>(y, x) = det / std::max(trace, 1e-8f);
      isotropyTemp.at<float>(y, x) = 4.0 * det / std::max(trace * trace, 1e-8f);
    }
  }
  weight = weightTemp.clone();
  isotropy = isotropyTemp.clone();
}

/**
 * @brief Finds Foerstner interest points in an image and returns their
 * location.
 * @param img The greyscale input image
 * @param sigmaGrad The standard deviation of the Gaussian kernel for the
 * directional gradients
 * @param sigmaNeighborhood The standard deviation of the Gaussian kernel for
 * computing the "neighborhood summation" of the structure tensor.
 * @param fractionalMinWeight Threshold on the weight as a fraction of the
 * mean of all locally maximal weights.
 * @param minIsotropy Threshold on the isotropy of interest points.
 * @returns List of interest point locations.
 */
std::vector<cv::Vec2i> getFoerstnerInterestPoints(const cv::Mat_<float> &img,
                                                  float sigmaGrad,
                                                  float sigmaNeighborhood,
                                                  float fractionalMinWeight,
                                                  float minIsotropy) {
  // TO DO !!!
  // create matrices
  cv::Mat_<float> gradY;
  cv::Mat_<float> gradX;
  cv::Mat_<float> A00;
  cv::Mat_<float> A11;
  cv::Mat_<float> A01;
  cv::Mat_<float> weights;
  cv::Mat_<float> isotropy;

  // compute directional gradients
  calculateDirectionalGradients(img, sigmaGrad, gradX, gradY);

  // compute structure tensor
  calculateStructureTensor(gradX, gradY, sigmaNeighborhood, A00, A01, A11);

  // compute weighst and shit
  calculateFoerstnerWeightIsotropy(A00, A01, A11, weights, isotropy);

  // compute threshold
  float thresh = fractionalMinWeight *
                 std::accumulate(weights.begin(), weights.end(), 0.0,
                                 [count = 0](double a, int b) mutable {
                                   return a + (b - a) / ++count;
                                 });

  std::vector<cv::Vec2i> result;
  for (int x = 0; x < weights.cols; x++) {
    for (int y = 0; y < weights.rows; y++) {
      if (isLocalMaximum(weights, x, y) && weights.at<float>(y, x) > thresh &&
          isotropy.at<float>(y, x) > minIsotropy) {
        result.push_back({x, y});
      }
    }
  }

  return result;
}

/* *****************************
  GIVEN FUNCTIONS
***************************** */

// Use this to compute kernel sizes so that the unit tests can simply hard
// checks for correctness.
unsigned getOddKernelSizeForSigma(float sigma) {
  unsigned kSize = (unsigned)std::ceil(5.0f * sigma) | 1;
  if (kSize < 3)
    kSize = 3;
  return kSize;
}

bool isLocalMaximum(const cv::Mat_<float> &weight, int x, int y) {
  for (int i = -1; i <= 1; i++)
    for (int j = -1; j <= 1; j++) {
      int x_ = std::min(std::max(x + j, 0), weight.cols - 1);
      int y_ = std::min(std::max(y + i, 0), weight.rows - 1);
      if (weight(y_, x_) > weight(y, x))
        return false;
    }
  return true;
}

} // namespace dip5
