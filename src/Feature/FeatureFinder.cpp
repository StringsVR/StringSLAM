#include "opencv2/core/mat.hpp"
#include "opencv2/core/types.hpp"
#include <memory>
#include <opencv2/video.hpp>
#include <StringSLAM/Feature/FeatureFinder.hpp>

namespace StringSLAM::Feature
{
    FeatureFinder::FeatureFinder(std::shared_ptr<OrbWrapper> orb_) : orb(orb_) {
        matcher = cv::BFMatcher::create(cv::NORM_HAMMING, false);
    }

    void FeatureFinder::getKeypoints(Frame &f) {
        // Clear and reserve descriptors and keypoints
        f.kp.clear();
        f.desc.reserve(orb->getMaxFeatures());

        // Detect and compute the frame and grab the kp and desc.
        orb->detectAndCompute(f.frame, f.kp, f.desc);
    }

    void FeatureFinder::drawKeypoint(Frame &f, cv::Mat &out, cv::Scalar color) {
        cv::drawKeypoints(
            f.frame,
            f.kp,
            out,
            color
        );
    }

    const std::vector<cv::DMatch> FeatureFinder::matchFrames(Frame &f1, Frame &f2) {
        // Clear Matches
        knnMatches.clear();
        matches.clear();

        // Match initial descriptors
        matcher->knnMatch(f1.desc, f2.desc, knnMatches, 2.0);
        matches.reserve(knnMatches.size() / 1.5f);

        // Filter out bad matches
        for (auto &m : knnMatches) {
            if (m.size() >= 2 && m[0].distance < 0.75f * m[1].distance) matches.push_back(m[0]);
        }

        // Return the good matches
        return matches;
    }

    const std::vector<cv::DMatch> FeatureFinder::matchFramesLK(Frame &f1, Frame &f2, cv::Size winSize, int maxLevel) {
        // Clear Matches (Knn isn't used so no need to clear it)
        matches.clear();

        // If keypoints for either frame don't exist just return pre-existing or null matches
        if (f1.kp.empty() || f2.kp.empty()) return matches;

        // Convert our keypoints into Point2f
        std::vector<cv::Point2f> pointsPrev, pointsNext;
        for (auto &kp : f1.kp) pointsPrev.push_back(kp.pt);

        std::vector<uchar> status;
        std::vector<float> err;

        cv::TermCriteria criteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);
        cv::calcOpticalFlowPyrLK(
            f1.frame,
            f2.frame,
            pointsPrev,
            pointsNext,
            status,
            err,
            winSize,
            maxLevel,
            criteria
        );

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

    void FeatureFinder::drawMatches(Frame &frame1, Frame &frame2, std::vector<cv::DMatch> matches, cv::Mat &out) {
        cv::drawMatches(frame1.frame, frame1.kp, frame2.frame, frame2.kp, matches, out);
    }

} // namespace StringSLAM::Feature
