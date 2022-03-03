# dip_group76_hw
This repository contains the homework exercises I did during the 2021/22 winter term for the digital image processing lecture at the Technical University of Berlin.

Note: the following run and test commands assume that you're on the 'build' folder created by cmake.

## Exercise 1
The idea of this exercise was to get acquanted with the OpenCV library, for this purpose we were tasked of processing an input image with a simple function. I implemented a function that corresponds to naive rotation of the image.

To run: 

    ./main [insert path to input image]
  
## Exercise 2
For this exercise we had to use various linear and non linear filtering techniques. The unit test benchmarks said techniques by means of a PSNR measurement.

To run: 
  
    ./main [insert path to input image]
    
To run test:

    ./unit_test [insert path ot input image]

## Exercise 3
In this exercise we were tasked with implementing and then testing the speed of different techniques for computing LTI filters. We also had the optional task of implementing unsharp masking.

To run: same as above.

To test: same as above.

## Exercise 4
Here, we were tasked with implementing a clipped inverse filter and a Wiener filter.

To run:

    ./main [insert path to image] [insert desired PSNR] [insert desired deviation for filter kernel]
    
## Exercise 5
In this exercise we were tasked with implementing a salient point detector using the Foerster interest point detector. 

To run:

    ./main [insert path to image] [insert desired deviation for gaussian directional derivatives]

I hope you find this interesting!.
    
