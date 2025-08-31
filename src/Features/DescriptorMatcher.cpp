#include <StringSLAM/Features/DescriptorMatcher.hpp>
#include <opencv2/video.hpp>

namespace StringSLAM::Features {
    const std::vector<cv::DMatch> DescriptorMatcher::matchFrames(Frame &f1, Frame &f2) {
        // Clear all lists
        knnMatches.clear();
        matches.clear();

        // Match init descriptors
        matcher->knnMatch(f1.desc, f2.desc, knnMatches, 2.0);
        matches.reserve(knnMatches.size() / 1.5f);  // Reserve 75% of space from knnMatchs to matches

        // Filter out bad matches
        for (auto &m : knnMatches) {
            if (m.size() >= 2 && m[0].distance < 0.75f * m[1].distance) matches.push_back(m[0]);
        }

        // Return the good matches
        return matches;
    }

    const std::vector<cv::DMatch> DescriptorMatcher::matchFramesLK(Frame &f1, Frame &f2) {
        // Clear all lists
        matches.clear();

        if (f1.kp.empty()) return matches;

        // Convert keypoints to Point2f
        std::vector<cv::Point2f> pointsPrev, pointsNext;
        for (auto &kp : f1.kp) pointsPrev.push_back(kp.pt);

        std::vector<uchar> status;
        std::vector<float> err;

        // Optical flow parameters
        cv::Size winSize(21, 21);
        int maxLevel = 3;  // Pyramid levels
        cv::TermCriteria criteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);

        cv::calcOpticalFlowPyrLK(f1.frame, f2.frame, pointsPrev, pointsNext, status, err, winSize, maxLevel, criteria);

        // Create matches from ssuccessfully tracked points
        for (size_t i = 0; i < pointsPrev.size(); i++) {
            if (status[i]) {
                cv::KeyPoint kpPrev(pointsPrev[i], 1.f);
                cv::KeyPoint kpNext(pointsNext[i], 1.f);
                f2.kp.push_back(kpNext);
                matches.push_back(cv::DMatch(static_cast<int>(i), static_cast<int>(f2.kp.size() - 1), 0.f));
            }
        }

        return matches;
    }

    void DescriptorMatcher::drawMatches(Frame &frame1, Frame &frame2, std::vector<cv::DMatch> matches, cv::Mat &out) {
        cv::drawMatches(frame1.frame, frame1.kp, frame2.frame, frame2.kp, matches, out);
    }
}