#include "opencv2/videoio.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include <StringSLAM/Tracker/MonoTracker.hpp>


namespace StringSLAM::Tracker {
    MonoTracker::MonoTracker(int id_, CameraModel cm_) : id(id_), cm(cm_) {
        bool opened = this->open();
        if (!opened) return;

        // Set the camera resolution after a succesfull camera open.
        cap.set(cv::CAP_PROP_FRAME_WIDTH, cm.capSize.width);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, cm.capSize.height);
    }

    void MonoTracker::read(Frame &f) {
        // Make sure camera is opened before capture.
        this->open();

        f.setTimestamp();
        cap.read(f.frame);
    }

    void MonoTracker::readUndistorted(Frame &f) {
        this->read(f);

        if (kD.empty()) {
            // If optimal K matrix is empty then create it and initialize undistortion map.
            kD = cv::getOptimalNewCameraMatrix(cm.cI.getK(), cm.cD.getD(), cm.capSize, 1.0);
            cv::initUndistortRectifyMap(cm.cI.getK(), cm.cD.getD(), cm.cD.R, kD, cm.capSize, CV_16FC1, m1, m2);
        }
        
        // Undistort frame using the optimized undistortion map
        cv::remap(f.frame, f.frame, m1, m2, cv::INTER_LINEAR);
    }
}