#pragma once
#include <StringSLAM/core.hpp>

namespace StringSLAM::Features
{
    class KeypointExtractor
    {
    private:
        std::shared_ptr<OrbWrapper> orb;
    public:
        KeypointExtractor(std::shared_ptr<OrbWrapper> orb_) : orb(orb_) {};
        ~KeypointExtractor() = default;

        // Get Keypoints from ORB->compute()
        void getKeypoints(Frame &frame);

        // Draw those keypoints to frame
        void drawKeypoints(Frame &frame, cv::Mat &out, cv::Scalar color);

        // Create Pointer Function
        static std::shared_ptr<KeypointExtractor> create(std::shared_ptr<OrbWrapper> orb_) {
            return std::make_shared<KeypointExtractor>(orb_);
        };
    };
        
} // namespace StringSLAM::Features