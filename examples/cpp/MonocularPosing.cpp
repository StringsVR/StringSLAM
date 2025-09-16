#include <StringSLAM/core.hpp>
#include <StringSLAM/Tracker/MonoTracker.hpp>
#include <StringSLAM/Feature/FeatureFinder.hpp>
#include <StringSLAM/Estimation/Poser/PoseEstimator2d.hpp>
#include <opencv2/core.hpp>
#include <future>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <deque>
#include <iostream>

using namespace StringSLAM;
using namespace cv;

int main() {
    try {
        std::shared_ptr<Accelerator> accelerator = Accelerator::create();
        // If GPU is optional, skip creating it for now.
        // accelerator->createGPU(true, 10);
        std::cout << "[INFO] Accelerator created successfully\n";
    } catch (const std::exception &e) {
        std::cerr << "[WARN] Accelerator init threw: " << e.what() << "\n";
    } catch (...) {
        std::cerr << "[WARN] Accelerator init threw unknown\n";
    }

    // -------------------------------
    // 1. Configure the camera 
    // -------------------------------
    CameraModel cm(
        CameraIntrinsic(200.0, 200.0, 320.0, 240.0), // fx, fy, cx, cy (cx/cy centered for 640x480)
        CameraDistortion(0.0, 0.0, 0.0, 0.0, 0.0),  // start with zero distortion to be safe
        Size(640, 480)                              // image size
    );

    // -------------------------------
    // 2. Create a MonoTracker for capturing frames
    // -------------------------------
    std::shared_ptr<Tracker::MonoTracker> mt1 = Tracker::MonoTracker::create(0, cm);
    if (!mt1) {
        std::cerr << "[FATAL] Failed to create MonoTracker\n";
        return 1;
    }
    if (!mt1->open()) {
        std::cerr << "[WARN] mt1->open() returned false; camera may not be available\n";
    } else {
        std::cout << "[INFO] Camera opened\n";
    }

    // -------------------------------
    // 3. Initialize ORB/FeatureFinder (kept commented if not used)
    // -------------------------------
    std::shared_ptr<OrbWrapper> orb = OrbWrapper::create(300, 1.2f, 4, 30, 0, 2, cv::ORB::FAST_SCORE, 32, 30); 
    std::shared_ptr<Feature::FeatureFinder> featureFinder = Feature::FeatureFinder::create(orb);
    std::shared_ptr<Estimation::Poser::PoseEstimator2d> poseEstimator = Estimation::Poser::PoseEstimator2d::create();
    Estimation::Poser::Pose2D relPose;
    std::vector<Eigen::Vector2d> pts_prev, pts_curr;

    std::vector<cv::DMatch> matches;
    std::deque<Frame> frames;
    Frame tempFrame, prevFrame;
    Mat output;

    int id = 0;
    for(;;) {
        id += 1;

        // Read frame from camera
        mt1->read(tempFrame);
        tempFrame.id = id;

        // Guard: did we actually get pixels?
        if (tempFrame.frame.empty()) {
            std::cerr << "[WARN] Empty frame grabbed. Skipping iteration.\n";
            // small sleep to avoid a hot loop when camera fails
            cv::waitKey(10);
            continue;
        }

        // If the frame list isn't empty then get prev frame.
        if (!frames.empty()) prevFrame = frames.back();

        output = tempFrame.frame.clone(); // clone to avoid aliasing issues

        // Safety checks before any feature operations
        if (featureFinder) featureFinder->getKeypoints(tempFrame);

        if (!prevFrame.kp.empty() && !tempFrame.kp.empty()) {
            matches = featureFinder->matchFramesLK(prevFrame, tempFrame);
        }

        if (!matches.empty() && !prevFrame.kp.empty()) {
            std::future<bool> poseFuture;
            featureFinder->drawMatches(prevFrame, tempFrame, matches, output);

            if (id % 30 == 0) {
                pts_prev.clear();
                pts_curr.clear();
                for (auto &m : matches) {
                    const auto &p = prevFrame.kp[m.queryIdx].pt;
                    const auto &q = tempFrame.kp[m.trainIdx].pt;
                    pts_prev.emplace_back(p.x, p.y);
                    pts_curr.emplace_back(q.x, q.y);
                }

                poseFuture = std::async(std::launch::async, [&]{
                    return poseEstimator->solvePose2D_GN(pts_prev, pts_curr, relPose);
                });

                std::cout << "[POSE] Δx=" << relPose.pos.x()
                    << " Δy=" << relPose.pos.y()
                    << " Δθ=" << relPose.pos.z() 
                    << "\n";
            }
    }

        // show only if not empty
        if (!output.empty()) {
            cv::imshow("Viewer", output);
        }

        int k = cv::waitKey(1);
        if (k == 27) break; // Esc to quit

        // Only add frame if it has keypoints (or at least image data)
        if (!tempFrame.kp.empty())
            frames.push_back(tempFrame);
        else
            ; // nothing

        // If more than 45 frames remove the first one
        if (frames.size() > 45)
            frames.pop_front();
    }

    return 0;
}
