#include <StringSLAM/Features/KeypointExtractor.hpp>
#include <opencv2/core/types.hpp>

namespace StringSLAM::Features {
    void KeypointExtractor::getKeypoints(Frame &frame) {
        frame.kp.clear();   // Clear the prexisting KeyPoints, otherwise lots of slow downs.
        frame.kp.reserve(orb->getMaxFeatures());    // Reserve KeyPoints for max features.

        // Detect and compute the frame and grab keypoints from our ORB wrapper.
        orb->detectAndCompute(frame.frame, frame.kp, frame.desc);
    }

    // This one really doesn't need explaining
    void KeypointExtractor::drawKeypoints(Frame &frame, cv::Mat &out, cv::Scalar color) {
        cv::drawKeypoints(
            frame.frame,
            frame.kp,
            out,
            color
        );
    }
}