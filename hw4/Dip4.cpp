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

std::sin, std::cos, std::tan, std::sqrt, std::pow, std::exp, .... all work as expected

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
    cv::Mat_<std::complex<float>> res;
    cv::dft(input, res, cv::DFT_COMPLEX_OUTPUT);
    return res;
}

/**
 * @brief Computes the real valued inverse DFT of a complex valued input
 * @param input Complex valued input, each pixel storing real and imaginary parts
 * @return Real valued output
 */
cv::Mat_<float> IDFTComplex2Real(const cv::Mat_<std::complex<float>> &input) {
    cv::Mat_<float> res;
    cv::dft(input, res, cv::DFT_INVERSE + cv::DFT_SCALE + cv::DFT_REAL_OUTPUT);
    return res;
}

/**
 * @brief Performes a circular shift in (dx,dy) direction
 * @param in Input matrix
 * @param dx Shift in x-direction
 * @param dy Shift in y-direction
 * @return Circular shifted matrix
 */
cv::Mat_<float> circShift(const cv::Mat_<float> &in, int dx, int dy) {
    assert(std::abs(dx) < in.rows && std::abs(dy) < in.cols);
    auto wrap_mod = [](int x, int d, int m) { return (x + d + m) % m; };
    cv::Mat_<float> res(in.size());
    for (int i = 0; i < in.rows; i++) {
        for (int j = 0; j < in.cols; j++) {
            res.at<float>(wrap_mod(i, dx, in.rows), wrap_mod(j, dy, in.cols)) = in.at<float>(i, j);
        }
    }
    return res;
}

/**
 * @brief Computes the thresholded inverse filter
 * @param input Blur filter in frequency domain (complex valued)
 * @param eps Factor to compute the threshold (relative to the max amplitude)
 * @return The inverse filter in frequency domain (complex valued)
 */
cv::Mat_<std::complex<float>> computeInverseFilter(const cv::Mat_<std::complex<float>> &input, const float eps) {
    float max = 0.;
    for (auto e : input) {
        max = std::max(max, std::abs(e));
    }

    auto t = eps * max;
    auto res = input.clone();
    for (auto &e : res) {
        e = 1.f / (std::abs(e) >= t ? e : t);
    }

    return res;
}

/**
 * @brief Applies a filter (in frequency domain)
 * @param input Image in frequency domain (complex valued)
 * @param filter Filter in frequency domain (complex valued), same size as input
 * @return The filtered image, complex valued, in frequency domain
 */
cv::Mat_<std::complex<float>> applyFilter(const cv::Mat_<std::complex<float>> &input, const cv::Mat_<std::complex<float>> &filter) {
    cv::Mat_<std::complex<float>> res;
    cv::mulSpectrums(input, filter, res, 0);
    return res;
}

cv::Mat_<float> restoreImageWithFilter(const cv::Mat_<float> &degraded, const cv::Mat_<float> &filter,
                                       std::function<cv::Mat_<std::complex<float>>(cv::Mat_<std::complex<float>>)> inverse_filter) {
    // pad and shft kernel
    cv::Mat_<float> tmp = cv::Mat_<float>::zeros(degraded.rows, degraded.cols);
    filter.copyTo(tmp(cv::Rect(0, 0, filter.cols, filter.rows)));
    auto filter_s = circShift(tmp, -filter.rows / 2, -filter.cols / 2);

    // compute spectra
    auto filter_spec = DFTReal2Complex(filter_s);
    auto image_spec = DFTReal2Complex(degraded);

    // compute inverse filter
    auto filter_inv = inverse_filter(filter_spec);

    // apply filter
    return IDFTComplex2Real(applyFilter(image_spec, filter_inv));
}

/**
 * @brief Function applies the inverse filter to restorate a degraded image
 * @param degraded Degraded input image
 * @param filter Filter which caused degradation
 * @param eps Factor to compute the threshold (relative to the max amplitude)
 * @return Restorated output image
 */
cv::Mat_<float> inverseFilter(const cv::Mat_<float> &degraded, const cv::Mat_<float> &filter, const float eps) {
    return restoreImageWithFilter(degraded, filter, [=](cv::Mat_<std::complex<float>> filter) { return computeInverseFilter(filter, eps); });
}

/**
 * @brief Computes the Wiener filter
 * @param input Blur filter in frequency domain (complex valued)
 * @param snr Signal to noise ratio
 * @return The wiener filter in frequency domain (complex valued)
 */
cv::Mat_<std::complex<float>> computeWienerFilter(const cv::Mat_<std::complex<float>> &input, const float snr) {
    std::complex<float> t = 1. / (snr * snr);
    auto res = input.clone();
    for (auto &e : res) {
        e = std::conj(e) / (std::norm(e) + t);
    }
    return res;
}

/**
 * @brief Function applies the wiener filter to restore a degraded image
 * @param degraded Degraded input image
 * @param filter Filter which caused degradation
 * @param snr Signal to noise ratio of the input image
 * @return Restored output image
 */
cv::Mat_<float> wienerFilter(const cv::Mat_<float> &degraded, const cv::Mat_<float> &filter, float snr) {
    return restoreImageWithFilter(degraded, filter, [=](cv::Mat_<std::complex<float>> filter) { return computeWienerFilter(filter, snr); });
}

/* *****************************
  GIVEN FUNCTIONS
***************************** */

/**
 * function degrades the given image with gaussian blur and additive gaussian noise
 * @param img Input image
 * @param degradedImg Degraded output image
 * @param filterDev Standard deviation of kernel for gaussian blur
 * @param snr Signal to noise ratio for additive gaussian noise
 * @return The used gaussian kernel
 */
cv::Mat_<float> degradeImage(const cv::Mat_<float> &img, cv::Mat_<float> &degradedImg, float filterDev, float snr) {

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
            kernels.at<float>((i - dy + img.rows) % img.rows, (j - dx + img.cols) % img.cols) = gaussKernel.at<float>(i, j);
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
