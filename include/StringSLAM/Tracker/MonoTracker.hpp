#pragma once

#include "StringSLAM/core.hpp"
#include <opencv2/videoio.hpp>

namespace StringSLAM::Tracker
{
    /**
     * @brief A tracking object that works as a camera and returns a Frame
     */
    class MonoTracker
    {
    private:
        int id;        
        CameraModel cm;

        // -- Below are private variables not specified but used in class. --
        //  OpenCV's object for camera capture
        cv::VideoCapture cap;

        // Optimized maps for undistortion in real time.
        cv::Mat m1, m2, kD;
    public:
        /**
         * @brief Constructs a MonoTracker
         * @param id_ ID of camera index
         * @param cm_ CameraModel for tracker.
         */
        MonoTracker(int id_, CameraModel cm_);
        ~MonoTracker() = default;
        
        /**
         * @brief Open camera tracker.
         * @return Camera opened succesfully
         */
        bool open(int API_PREF=cv::CAP_V4L2);

        /// Release camera
        inline void release() {
            cap.release();
        }

        /**
         * @brief Get Frame from capture
         * @return Timestamped and captured frame
         */
        void read(Frame &f);

        /**
         * @brief Get undistorted frame based off CameraModel
         * @return Undistorted Frame
         */
        void readUndistorted(Frame &f);
        
        /**
         * @brief Get FPS of tracker camera
         * @return FPS of capture device
         */
        inline double getFPS() { return cap.get(cv::CAP_PROP_FPS); }

        /**
         * @brief Get CameraModel Object
         * @return CameraModel
         */
        inline CameraModel getCameraModel() { return cm; }

        /**
         * @brief Create Shared Pointer of MonoTracker object
         * @return Shared Pointer of MonoTracker
         */
        static std::shared_ptr<MonoTracker> create(int id_, CameraModel cm_) {
            return std::make_shared<MonoTracker>(id_, cm_);
        }
    };
    
} // namespace StringSLAM::Feature
