#include "opencv2/core/mat.hpp"
#include "opencv2/core/types.hpp"
#include <memory>
#include <opencv2/video.hpp>
#include <StringSLAM/Feature/FeatureFinder.hpp>

namespace StringSLAM::Feature
{
    FeatureFinder::FeatureFinder(std::shared_ptr<OrbWrapper> &orb_) : orb(orb_) {
        matcher = cv::BFMatcher::create(cv::NORM_HAMMING, false);
    }

    void FeatureFinder::getKeypoints(Frame &f) {
        // Clear and reserve descriptors and keypoints
        f.kp.clear();
        f.desc.release();

        if (f.frame.empty())
            return;

        // Detect and compute the frame and grab the kp and desc.
        orb->detectAndCompute(f.frame, f.kp, f.desc);
    }

    void FeatureFinder::drawKeypoint(Frame &f, cv::Mat &out, cv::Scalar &color) {
        cv::drawKeypoints(
            f.frame,
            f.kp,
            out,
            color
        );
    }

    const std::vector<cv::DMatch> FeatureFinder::matchFramesLK(
        Frame &f1,
        Frame &f2,
        cv::Size winSize,
        int maxLevel
    ) {
        matches.clear();

        // If keypoints or descriptors are missing, return empty
        if (f1.kp.empty() || f2.kp.empty())
            return matches;

        // Convert keypoints from f1 into Point2f
        std::vector<cv::Point2f> pointsPrev;
        pointsPrev.reserve(f1.kp.size());
        for (auto &kp : f1.kp) pointsPrev.push_back(kp.pt);

        std::vector<cv::Point2f> pointsNext;
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

        // Build matches for successfully tracked points
        for (size_t i = 0; i < pointsPrev.size(); i++) {
            if (status[i]) {
                // Find nearest keypoint in f2 for the tracked location
                // (brute force since f2.kp is usually not huge)
                int bestIdx = -1;
                float bestDist = std::numeric_limits<float>::max();

                for (size_t j = 0; j < f2.kp.size(); j++) {
                    float dx = f2.kp[j].pt.x - pointsNext[i].x;
                    float dy = f2.kp[j].pt.y - pointsNext[i].y;
                    float dist = dx*dx + dy*dy;
                    if (dist < bestDist) {
                        bestDist = dist;
                        bestIdx = static_cast<int>(j);
                    }
                }

                if (bestIdx >= 0 && bestIdx < static_cast<int>(f2.kp.size())) {
                    cv::DMatch m;
                    m.queryIdx = static_cast<int>(i);     // index into frame1.kp
                    m.trainIdx = bestIdx;                 // index into frame2.kp
                    m.imgIdx   = 0;                       // (unused here)
                    m.distance = std::sqrt(bestDist);     // optional: keep it real distance
                    matches.push_back(m);
                }
            }
        }

        return matches;
    }


    void FeatureFinder::drawMatches(Frame &frame1, Frame &frame2, std::vector<cv::DMatch> &matches, cv::Mat &out) {
        matches.erase(
    std::remove_if(matches.begin(), matches.end(),
        [&](const cv::DMatch &m) {
            return m.queryIdx < 0 || m.queryIdx >= static_cast<int>(frame1.kp.size()) ||
                   m.trainIdx < 0 || m.trainIdx >= static_cast<int>(frame2.kp.size());
        }),
    matches.end()
    );


        cv::drawMatches(frame1.frame, frame1.kp, frame2.frame, frame2.kp, matches, out);
    }

} // namespace StringSLAM::Feature
