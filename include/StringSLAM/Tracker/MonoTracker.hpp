#pragma once
#include "StringSLAM/core.hpp"
#include <memory>
#include <opencv2/core/mat.hpp>
#include <opencv2/videoio.hpp>

namespace StringSLAM::Tracker {
    class MonoTracker
    {
    private:
        cv::VideoCapture cap;
        cv::Mat m1, m2; // Remapped matrix for fast undistortion
        cv::Mat Kd;     // Optimal K from D
        CameraModel cm; // Camera Spec

        int id;         // Camera ID
    public:
        MonoTracker(int id_, CameraModel &cm_);
        ~MonoTracker();

        bool open();
        void release();

        void read(Frame &frame);
        void readUndistorted(Frame &frame);

        inline double getFPS() { return cap.get(cv::CAP_PROP_FPS); }

        static std::shared_ptr<MonoTracker> create(int id_, CameraModel cm_) {
            return std::make_shared<MonoTracker>(id_, cm_);
        }
    };
}