#pragma once

#include "StringSLAM/Tracker/MonoTracker.hpp"
namespace StringSLAM::Tracker
{
    /**
     * @brief A tracking object classified as a "Stereo Tracker" created by 2 MonoTrackers.
     */
    class StereoTracker
    {
    private:
        // Specified Trackers that create Stereo
        MonoTracker mt1, mt2;

        // Stereo Distortion
        StereoCameraDistortion scd;

        // SGBM Matcher for stereo computation, SBM is also a option but thats TODO.
        StereoSGBMWrapper sgbm;

        // -- Below are private variables not specified but used in class. --
        // Rectified Stereo correction map
        cv::Mat m1x, m1y, m2x, m2y;
    public:
        /**
         * @brief Construct StereoTracker from parameters
         * @param mt1_ MonoTracker
         * @param mt2_ MonoTracker
         * @param scd Stereo camera distortion structure
         */
        StereoTracker(MonoTracker &mt1_, MonoTracker &mt2_, StereoCameraDistortion &scd_, StereoSGBMWrapper &sgbm_);
        ~StereoTracker() = default;

        /**
         * @brief Open both mono trackers.
         * @return Cameras opened succesfully
         */
        inline bool open() {
            return (mt1.open() && mt2.open());
        }

        /// Release cameras
        inline void release() {
            mt1.release();
            mt2.release();
        }

        /**
         * @brief Get StereoFrame from capture
         * @return Timestamped frame.
         */
        void read(StereoFrame &sf);

        /**
         * @brief Read depthMap from 2 MonoTracker captures.
         * @return Timestamped, depth calculated frame.
         */
        void readDepth(StereoFrame &sf);

        /**
         * @brief Create Shared Pointer of StereoTracker object
         * @return Shared Pointer of StereoTracker
         */
        static std::shared_ptr<StereoTracker> create(MonoTracker mt1_, MonoTracker mt2_, StereoCameraDistortion scd_) {
            return std::make_shared<StereoTracker>(mt1_, mt2_, scd_);
        }
    };
    
} // namespace StringSLAM::Tracker
