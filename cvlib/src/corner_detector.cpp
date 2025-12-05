/* FAST corner detector algorithm implementation.
 * @file
 * @date 2018-10-16
 * @author Anonymous
 */

#include "cvlib.hpp"

#include <ctime>

namespace cvlib
{
// static
cv::Ptr<corner_detector_fast> corner_detector_fast::create()
{
    return cv::makePtr<corner_detector_fast>();
}

void corner_detector_fast::detect(cv::InputArray image, CV_OUT std::vector<cv::KeyPoint>& keypoints, cv::InputArray /*mask = cv::noArray()*/)
{
    keypoints.clear();
    // \todo implement FAST with minimal LOCs(lines of code), but keep code readable.
}

// void corner_detector_fast::compute(cv::InputArray, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors)
// {
//     std::srand(unsigned(std::time(0))); // \todo remove me
//     // \todo implement any binary descriptor
//     const int desc_length = 2;
//     descriptors.create(static_cast<int>(keypoints.size()), desc_length, CV_32S);
//     auto desc_mat = descriptors.getMat();
//     desc_mat.setTo(0);

//     int* ptr = reinterpret_cast<int*>(desc_mat.ptr());
//     for (const auto& pt : keypoints)
//     {
//         for (int i = 0; i < desc_length; ++i)
//         {
//             *ptr = std::rand();
//             ++ptr;
//         }
//     }
// }


void corner_detector_fast::compute(cv::InputArray image, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors)
{
    cv::Mat img = image.getMat();

    cv::Mat gray;
    if (img.channels() == 3) {
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = img;
    }

    const int desc_length = 256;
    const int patch_size = 31;
    const int patch_radius = patch_size / 2;

    descriptors.create(static_cast<int>(keypoints.size()), desc_length / 8, CV_8U);
    cv::Mat desc_mat = descriptors.getMat();
    desc_mat.setTo(0);

    cv::Mat smoothed;
    cv::GaussianBlur(gray, smoothed, cv::Size(5, 5), 2.0, 2.0);

    std::vector<cv::Point2i> pattern_points_A(desc_length);
    std::vector<cv::Point2i> pattern_points_B(desc_length);

    std::srand(42);
    for (int i = 0; i < desc_length; ++i) {
        float angle1 = static_cast<float>(std::rand()) / RAND_MAX * 2 * CV_PI;
        float radius1 = static_cast<float>(std::rand()) / RAND_MAX * patch_radius;
        pattern_points_A[i] = cv::Point2i(
            static_cast<int>(radius1 * std::cos(angle1)),
            static_cast<int>(radius1 * std::sin(angle1))
        );

        float angle2 = static_cast<float>(std::rand()) / RAND_MAX * 2 * CV_PI;
        float radius2 = static_cast<float>(std::rand()) / RAND_MAX * patch_radius;
        pattern_points_B[i] = cv::Point2i(
            static_cast<int>(radius2 * std::cos(angle2)),
            static_cast<int>(radius2 * std::sin(angle2))
        );
    }

    for (size_t k = 0; k < keypoints.size(); ++k) {
        const cv::KeyPoint& kp = keypoints[k];
        int x = static_cast<int>(kp.pt.x + 0.5f);
        int y = static_cast<int>(kp.pt.y + 0.5f);

        if (x - patch_radius < 0 || x + patch_radius >= gray.cols ||
            y - patch_radius < 0 || y + patch_radius >= gray.rows) {
            desc_mat.row(static_cast<int>(k)).setTo(0);
            continue;
        }

        uchar* desc_ptr = desc_mat.ptr(static_cast<int>(k));

        for (int i = 0; i < desc_length; ++i) {
            int x1 = x + pattern_points_A[i].x;
            int y1 = y + pattern_points_A[i].y;

            int x2 = x + pattern_points_B[i].x;
            int y2 = y + pattern_points_B[i].y;

            if (x1 < 0 || x1 >= gray.cols || y1 < 0 || y1 >= gray.rows ||
                x2 < 0 || x2 >= gray.cols || y2 < 0 || y2 >= gray.rows) {
                continue;
            }

            uchar intensity1 = smoothed.at<uchar>(y1, x1);
            uchar intensity2 = smoothed.at<uchar>(y2, x2);

            bool bit_value = (intensity1 < intensity2);

            if (bit_value) {
                int byte_index = i / 8;
                int bit_index = i % 8;
                desc_ptr[byte_index] |= (1 << bit_index);
            }
        }
    }
}




void corner_detector_fast::detectAndCompute(cv::InputArray, cv::InputArray, std::vector<cv::KeyPoint>&, cv::OutputArray descriptors, bool /*= false*/)
{
    // \todo implement me
}
} // namespace cvlib
