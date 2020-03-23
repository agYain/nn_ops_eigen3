#include <iostream>
#include <cmath>
#include <unsupported/Eigen/CXX11/Tensor>


/*
* @para input:        4D input tensor -- Ci, H, W, B
* @para kernels:      4D kernels tensor -- Co, Ci, kH, kW
* @para stride:       size 2 array -- {row_stride, col_stride}
* @para dilation:     size 2 array -- {row_dilation, col_dilation}
* @para padding_type: padding type
*
* @return:            4D output tensor -- Co, Ho, Wo, B
*/
template<typename T>
Eigen::Tensor<T, 4> conv2d(const Eigen::Tensor<T, 4>& input,
                           const Eigen::Tensor<T, 4>& kernels,
                           const Eigen::array<int, 2>& stride = Eigen::array<int, 2>({1, 1}),
                           const Eigen::array<int, 2>& dilation = Eigen::array<int, 2>({1, 1}),
                           const Eigen::PaddingType padding_type = Eigen::PADDING_SAME)
{
    int numChannels  = input.dimension(0);
    int inputRows    = input.dimension(1);
    int inputCols    = input.dimension(2);
    int numBatches   = input.dimension(3);

    int numKernels   = kernels.dimension(0);
    int kernelRows   = kernels.dimension(2);
    int kernelCols   = kernels.dimension(3);

    int kernelRowsEffective = kernelRows + (kernelRows - 1) * (dilation[0] - 1);
    int kernelColsEffective = kernelCols + (kernelCols - 1) * (dilation[1] - 1);

    int outputHeight, outputWidth;
    if (padding_type == Eigen::PADDING_VALID)
    { 
        outputHeight = ceil( (inputRows - kernelRowsEffective + 1.f) / static_cast<float>(stride[0]) );
        outputWidth  = ceil( (inputCols - kernelColsEffective + 1.f) / static_cast<float>(stride[1]) );
    }
    else
    {
        outputHeight = ceil( inputRows / static_cast<float>(stride[0]) );
        outputWidth  = ceil( inputCols / static_cast<float>(stride[1]) );
    }
    
    Eigen::array<int, 3> patchMatrixDims({numChannels*kernelRows*kernelCols, outputHeight*outputWidth, numBatches});
    Eigen::array<int, 2> kernelMatrixDims({numKernels, numChannels*kernelRows*kernelCols});
    Eigen::array<int, 4> outputDims({numKernels, outputHeight, outputWidth, numBatches});

    Eigen::array<Eigen::IndexPair<int>, 1> contractDims = {Eigen::IndexPair<int>(1, 0)};

    
    return kernels.reshape(kernelMatrixDims)
                  .contract(input.extract_image_patches(kernelRows, kernelCols, 
                                                        stride[0], stride[1], 
                                                        dilation[0], dilation[1], 
                                                        padding_type)
                                 .reshape(patchMatrixDims),
                            contractDims)
                  .reshape(outputDims);
}


/*
* @para input:        4D input tensor -- Ci, H, W, B
* @para kernelSize:   size 2 array -- {kernel_height, kernel_width}
* @para stride:       size 2 array -- {row_stride, col_stride}
* @para dilation:     size 2 array -- {row_dilation, col_dilation}
* @para padding_type: padding type
*
* @return:            4D output tensor -- Co, Ho, Wo, B
*/
template<typename T>
Eigen::Tensor<T, 4> maxpooling2d(const Eigen::Tensor<T, 4>& input,
                                 const Eigen::array<int, 2>& kernelSize,
                                 const Eigen::array<int, 2>& stride = Eigen::array<int, 2>({0, 0}),
                                 const Eigen::array<int, 2>& dilation = Eigen::array<int, 2>({1, 1}),
                                 const Eigen::PaddingType padding_type = Eigen::PADDING_VALID)
{
    int numChannels  = input.dimension(0);
    int inputRows    = input.dimension(1);
    int inputCols    = input.dimension(2);
    int numBatches   = input.dimension(3);

    int kernelRows   = kernelSize[0];
    int kernelCols   = kernelSize[1];

    int kernelRowsEffective = kernelRows + (kernelRows - 1) * (dilation[0] - 1);
    int kernelColsEffective = kernelCols + (kernelCols - 1) * (dilation[1] - 1);

    int strideRows = (!stride[0]) ? kernelRows : stride[0];
    int strideCols = (!stride[1]) ? kernelCols : stride[1];

    int outputHeight, outputWidth;
    if (padding_type == Eigen::PADDING_VALID)
    { 
        outputHeight = ceil( (inputRows - kernelRowsEffective + 1.f) / static_cast<float>(strideRows) );
        outputWidth  = ceil( (inputCols - kernelColsEffective + 1.f) / static_cast<float>(strideCols) );
    }
    else
    {
        outputHeight = ceil( inputRows / static_cast<float>(strideRows) );
        outputWidth  = ceil( inputCols / static_cast<float>(strideCols) );
    }
    
    Eigen::array<int, 4> outputDims({numChannels, outputHeight, outputWidth, numBatches});

    Eigen::array<int, 2> reduceDims({1, 2});

    return input.extract_image_patches(kernelRows, kernelCols,
                                       strideRows, strideCols,
                                       dilation[0], dilation[1],
                                       padding_type)
                .maximum(reduceDims)
                .reshape(outputDims);
}


template<typename T>
Eigen::Tensor<T, 4> addBias(const Eigen::Tensor<T, 4>& input,
                            const Eigen::Tensor<T, 1>& bias)
{
    Eigen::Tensor<T, 4> output(input.dimensions());
    Eigen::DSizes<int, 1> oneDimSize(input.size());
    Eigen::DSizes<int, 1> broadcastSize(input.size()/bias.dimension(0));

    output.reshape(oneDimSize) 
        = input.reshape(oneDimSize) + bias.broadcast(broadcastSize).reshape(oneDimSize);
    
    return output;
}


template<typename T, int R>
Eigen::Tensor<T, R> ReLU(Eigen::Tensor<T, R>& input)
{
    return input.unaryExpr([](T e){return std::max(static_cast<T>(0), e);});
}
