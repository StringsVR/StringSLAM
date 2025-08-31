#include <opencv2/core/types.hpp>
#include <StringSLAM/Tracker/MonoTracker.hpp>
#include <StringSLAM/Features/KeypointExtractor.hpp>
#include <StringSLAM/Features/DescriptorMatcher.hpp>
#include <StringSLAM/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv4/opencv2/features2d.hpp>
#include <deque>

using namespace StringSLAM;

int main() {
    std::shared_ptr<Tracker::MonoTracker> mt1 = Tracker::MonoTracker::create(0, CameraModel());

    std::shared_ptr<Accelerator> accelerator = Accelerator::create();
    accelerator->createGPU(true, 10);

    std::shared_ptr<OrbWrapper> orb = OrbWrapper::create(800, 1.2f, 4, 30, 0, 2, cv::ORB::HARRIS_SCORE, 32, 30);
    std::shared_ptr<Features::KeypointExtractor> kpExtractor = Features::KeypointExtractor::create(orb);
    std::shared_ptr<Features::DescriptorMatcher> descMatcher = Features::DescriptorMatcher::create(orb);

    mt1->open();

    std::vector<cv::DMatch> matches;
    std::deque<Frame> frames;
    Frame tempFrame, prevFrame;
    cv::Mat out;

    for (;;) {
        // If the frame list isn't empty then get prev frame.
        if (!frames.empty()) prevFrame = frames.back();
        
        // Read frame from camera
        mt1->read(tempFrame);
        out = tempFrame.frame;

        // Get keypoints
        kpExtractor->getKeypoints(tempFrame);
        if (!prevFrame.kp.empty()) {
            // Get matches for keypoints and generator descriptor
            matches = descMatcher->matchFramesLK(tempFrame, prevFrame);

            // Draw matches
            if (!matches.empty()) descMatcher->drawMatches(tempFrame, prevFrame, matches, out);
        }

        cv::imshow("Viewer", out);
        cv::waitKey(mt1->getFPS());

        // Only add frame if it has keypoints
        if (!tempFrame.kp.empty())
            frames.push_back(tempFrame);

        // If more than 45 frame remove the first one
        if (frames.size() > 45) 
            frames.pop_front();
    }
}