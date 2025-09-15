#include "opencv2/calib3d.hpp"
#include <StringSLAM/Tracker/StereoTracker.hpp>
#include <opencv2/imgproc.hpp>

namespace StringSLAM::Tracker {
    StereoTracker::StereoTracker(MonoTracker &mt1_, MonoTracker &mt2_, StereoCameraDistortion &scd_, StereoSGBMWrapper &sgbm_) : 
        mt1(mt1_), mt2(mt2_), scd(scd_), sgbm(sgbm_) { }

    void StereoTracker::read(StereoFrame &sf) {
        // Make sure camera is opened before capture.
        this->open();

        sf.setTimestamp();
        mt1.read(sf.frameLeft);
        mt2.read(sf.frameRight);
    }

    void StereoTracker::readDepth(StereoFrame &sf) {
        this->read(sf);

        if (scd.Q.empty()) {
            // Rectify stereo and save results to a StereoCameraDistortion object.
            cv::stereoRectify(
                mt1.getCameraModel().cI.getK(), 
                mt1.getCameraModel().cD.getD(), 
                mt2.getCameraModel().cI.getK(), 
                mt2.getCameraModel().cD.getD(),
                mt1.getCameraModel().capSize, 
                scd.R, 
                scd.T, 
                scd.R1, 
                scd.R2, 
                scd.P1, 
                scd.P2, 
                scd.Q
            );

            // Calculate rectify maps for left frame, right frame.
            cv::initUndistortRectifyMap(
                mt1.getCameraModel().cI.getK(),
                mt1.getCameraModel().cD.getD(), 
                scd.R1,
                scd.P1,
                mt1.getCameraModel().capSize,
                CV_32FC1,
                m1x,
                m1y
            );
            
            cv::initUndistortRectifyMap(
                mt2.getCameraModel().cI.getK(),
                mt2.getCameraModel().cD.getD(), 
                scd.R2,
                scd.P2,
                mt2.getCameraModel().capSize,
                CV_32FC1,
                m2x,
                m2y
            );
        }

        // Undistort capture frames from pre-calculated rectify maps
        cv::remap(sf.frameLeft.frame, sf.frameLeft.frame, m1x, m1y, cv::INTER_LINEAR);
        cv::remap(sf.frameRight.frame, sf.frameRight.frame, m2x, m2y, cv::INTER_LINEAR);

        sgbm.compute(sf);
    }
}