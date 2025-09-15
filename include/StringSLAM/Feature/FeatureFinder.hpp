#pragma once

#include "StringSLAM/core.hpp"
namespace StringSLAM::Feature
{
    /**
     * @brief Class meant for handling point detection.
     * 
     * A Class Utility meant to be used to find keypoints and descriptors in a frame.
     * Also, to match descriptors between 2 frames
     */
    class FeatureFinder
    {
    private:
        // ORB object specified in contructor
        std::shared_ptr<OrbWrapper> orb;

        // -- Below are private variables not specified but used in class. --

        // ---- Description Matcher ----
        // Fast matching class provided by OpenCV
        cv::Ptr<cv::BFMatcher> matcher;

        // List of "KnnMatch", if your not familiar refer to google.
        std::vector<std::vector<cv::DMatch>> knnMatches;

        // List of matches filtered from Knn.
        std::vector<cv::DMatch> matches;

    public:
        /**
         * @brief Create constructor for FeatureFinder
         * @param orb_ OrbWrapper
         */
        FeatureFinder(std::shared_ptr<OrbWrapper> orb_);
        ~FeatureFinder();

        /**
         * @brief Get keypoints from frame
         * @return Frame with specified keypoints.
         */
        void getKeypoints(Frame &f);

        /**
         * @brief Draw keypoints to frame
         * @return Mat with keypoint
         */
        void drawKeypoint(Frame &f, cv::Mat &out, cv::Scalar color);

        /**
         * @brief Match descriptions from 2 frames
         * 
         * @param f1 Frame 1
         * @param f2 Frame 2
         */
        const std::vector<cv::DMatch> matchFrames(Frame &f1, Frame &f2);

        /**
         * @brief Match descriptors from 2 frames using Lucas–Kanade (speedy)
         * 
         * @param f1 Frame 1
         * @param f2 Frame 2
         * @param winSize ROI for matches.
         * @param maxLevel Amount of pyramid levels applied to frames
         */
        const std::vector<cv::DMatch> matchFramesLK(Frame &f1, Frame &f2, cv::Size winSize = cv::Size(21, 21), int maxLevel = 3);

        /**
         * @brief Create Shared Pointer of FeatureFinder object
         * @return Shared Pointer of FeatureFinder
         */
        static std::shared_ptr<FeatureFinder> create(std::shared_ptr<OrbWrapper> orb_) {
            return std::make_shared<FeatureFinder>(orb_);
        }
    };
} // namespace StringSLAM::Feature
