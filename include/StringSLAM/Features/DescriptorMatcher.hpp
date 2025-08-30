#pragma once
#include <StringSLAM/core.hpp>
#include <memory>

namespace StringSLAM::Features
{
    // Matches descriptors between 2 frames.
    class DescriptorMatcher
    {
        private:
            std::shared_ptr<OrbWrapper> orb;                    // Orb Wrapper
            cv::Ptr<cv::BFMatcher> matcher;                     // OpenCV DescriptorMatcher Class "BFMatcher", its quite fast
            std::vector<std::vector<cv::DMatch>> knnMatches;    // What are knnMatches?
            std::vector<cv::DMatch> matches;                    // Filtered Matches from knnMatches (these r the good ones)
        public:
            DescriptorMatcher(std::shared_ptr<OrbWrapper> orb_) 
                // We set it to "NORM_HAMMING" due to speed, there are other options though.
                : orb(orb_) { matcher = cv::BFMatcher::create(cv::NORM_HAMMING, false); };

            ~DescriptorMatcher() = default;

            // Default Descriptor Matching Method (Simple)
            const std::vector<cv::DMatch> matchFrames(Frame &f1, Frame &f2);

            // Faster Descriptor Matching Method (More Complex)
            const std::vector<cv::DMatch> matchFramesLK(Frame &f1, Frame &f2);

            // Draw the found matches
            void drawMatches(Frame &frame1, Frame &frame2, std::vector<cv::DMatch> matches, cv::Mat &out);

            // Create Pointer Function
            static std::shared_ptr<DescriptorMatcher> create(std::shared_ptr<OrbWrapper> orb_) {
                return std::make_shared<DescriptorMatcher>(orb_);
            };
    };
    
} // namespace StringSLAM::Features