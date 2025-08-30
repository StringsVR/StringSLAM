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

        void getKeypoints(Frame &frame);
        void drawKeypoints(Frame &frame, cv::Mat &out, cv::Scalar color);

        static std::shared_ptr<KeypointExtractor> create(std::shared_ptr<OrbWrapper> orb_) {
            return std::make_shared<KeypointExtractor>(orb_);
        };
    };
        
} // namespace StringSLAM::Features