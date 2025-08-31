#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <StringSLAM/Tracker/MonoTracker.hpp>
#include <opencv2/calib3d.hpp>

namespace StringSLAM::Tracker {
    MonoTracker::MonoTracker(int id_, CameraModel &cm_) : id(id_) {
        cm.copyFrom(cm_);
    }

    MonoTracker::~MonoTracker() {
        this->release();
    }

    bool MonoTracker::open() {
        return cap.open(id, cv::CAP_V4L2);
    }

    void MonoTracker::release() {
        cap.release();
    }

    void MonoTracker::read(Frame &frame) {
        cap.read(frame.frame);
        frame.timestamp = std::chrono::system_clock::now();
    }

    void MonoTracker::readUndistorted(Frame &frame) {
        this->read(frame);
        if (m1.empty() || m2.empty() || Kd.empty()) {
            Kd = cv::getOptimalNewCameraMatrix(cm.K, cm.D, cm.capSize, 1.0);
            cv::initUndistortRectifyMap(cm.K, cm.D, cm.R, Kd, cm.capSize, CV_16FC1, m1, m2);
        }

        cv::remap(frame.frame, frame.frame, m1, m2, cv::INTER_LINEAR);
    }
}