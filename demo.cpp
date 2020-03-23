#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>
#include <chrono>

#include "nn_ops.h" 


using namespace std::chrono;


int main()
{
    /*
    *  Image sensor: Ci, H, W, B
    */
    Eigen::Tensor<float, 4> image(3, 8, 1, 1); // Ci, H, W, B
    image.setValues(
        {
            { {{0.f}}, {{1.f}}, {{2.f}}, {{3.f}}, {{4.f}}, {{5.f}}, {{6.f}}, {{7.f}} },
            { {{1.f}}, {{2.f}}, {{3.f}}, {{4.f}}, {{5.f}}, {{6.f}}, {{7.f}}, {{8.f}} },
            { {{2.f}}, {{3.f}}, {{4.f}}, {{5.f}}, {{6.f}}, {{7.f}}, {{8.f}}, {{9.f}} },
        }
    );

    /*
    *  Kernels: Co, Ci, kH, kW
    */
    Eigen::Tensor<float, 4> kernels(2, 3, 2, 1);     // Co, Ci, Kh, Kw
    kernels.setValues(
        {
            { {{1.f}, {2.f}}, {{3.f}, {4.f}}, {{5.f}, {6.f}} },
            { {{6.f}, {5.f}}, {{4.f}, {3.f}}, {{2.f}, {1.f}} },
        }
    );

    Eigen::Tensor<float, 1> bias(2);
    bias.setValues({10, 20});

    Eigen::Tensor<float, 4> output;
    
    auto start = high_resolution_clock::now();

    // 2d convolution
    output = conv2d(image,
                    kernels, 
                    Eigen::array<int, 2>({2, 1}), 
                    Eigen::array<int, 2>({1, 1}), 
                    Eigen::PADDING_VALID);

    // add bias channel-wise
    output = addBias(output, bias);

    // element-wise ReLU
    output = ReLU(output);

    // 2d max-pooling
    output = maxpooling2d(output, Eigen::array<int, 2>({2, 1}));

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << "Processing time: " << duration.count() << "us" << std::endl;

    for (int i = 0; i < output.dimension(0); ++i)
    {
        for (int j = 0; j < output.dimension(1); ++j)
        {
            std::cout << output(i, j, 0, 0) << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
