#pragma once
#include "opencv2/calib3d.hpp"
#include "opencv2/core/mat.hpp"
#include "opencv2/core/types.hpp"
#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>

namespace StringSLAM
{
    // ------------------ Structs ------------------

    /**
     * @brief Stores the intrinsic parameters of a camera.
     *
     * The intrinsic parameters define the internal characteristics of the camera,
     * Such as focal lengths and principal point.
     */
    struct CameraIntrinsic
    {
        /// Focal length in x direction (pixels)
        double fx;

        /// Focal length in y direction (pixels)
        double fy;

        /// Principal point x-coordinate (pixels)
        double cx;

        /// Principal point y-coordinate (pixels)
        double cy;
        
        /**
         * @brief Constructs a CameraIntrinsic with given parameters.
         * @param fx_ Focal length in x direction
         * @param fy_ Focal length in y direction
         * @param cx_ Principal point x-coordinate
         * @param cy_ Principal point y-coordinate
         */
        CameraIntrinsic(const double fx_, 
            const double fy_, 
            const double cx_, 
            const double cy_
        ) : fx(fx_), fy(fy_), cx(cx_), cy(cy_) {};
        
        /**
         * @brief Default constructor. Initializes values to default (uninitialized).
         */
        CameraIntrinsic() = default;
        
        /**
         * @brief Returns the camera intrinsic matrix K (3x3).
         * @return cv::Mat The 3x3 intrinsic matrix
         */
        cv::Mat getK() const {
            static cv::Mat K = (cv::Mat_<double>(3,3) << fx,0,cx,0,fy,cy,0,0,1);
            return K;
        }

    };
    
    /**
     * @brief Stores the distortion parameters of a camera.
     * 
     * The distortion parameters define the distortional coefficients of a camera.
     * Such as radial or tangential.
     */
    struct CameraDistortion
    {
        /// First-Order radial distortion value
        double k1 = 0.0;

        /// Second-Order radial distortion value (Corrects K1 curvature)
        double k2 = 0.0;

        /// Third-Order radial distortion value (Fine tunes the distortion correction)
        double k3 = 0.0;
        
        /// Optional higher-order term for correct Fish-Eye lenses
        double k4 = 0.0;

        /// Optional higher-order term for correct Fish-Eye lenses
        double k5 = 0.0;

        /// Optional higher-order term for correct Fish-Eye lenses
        double k6 = 0.0;

        /// Tangential distortion across the X-axis
        double p1 = 0.0;
        
        /// Tangential distortion across the Y-axis
        double p2 = 0.0;

        /// Rectification rotation (Typically will be a identity 3x3 matrix in monocular)
        cv::Mat R = cv::Mat::eye(3, 3, CV_64F);

        /**
         * @brief Constructs a distortion profile within the given parameters.
         * @param k1_ First-Order radial distortion
         * @param k2_ Second-Order radial distortion
         * @param k3_ Third-Order radial distortion
         * @param p1_ X-Axis tangential distortion
         * @param p2_ Y-Axis tangential distortion
         */
        CameraDistortion(double k1_=0.0, double k2_=0.0, double k3_=0.0,
            double p1_=0.0, double p2_=0.0) : k1(k1_), k2(k2_), k3(k3_), p1(p1_), p2(p2_) {};
        
        /**
         * @brief Returns the camera distortion matrix D (1x8).
         * @return cv::Mat The 1x8 distortion matrix
         */
        cv::Mat getD() const {
            static cv::Mat D = (cv::Mat_<double>(1,8) << k1, k2, p1, p2,  k3, k4, k5, k6);
            return D;
        }
    };
    

    /**
     * @brief Full Camera Parameters for a MonoTracker instance.
     * (\ref StringSLAM::Tracker::MonoTracker).
     * 
     * @see CameraIntrinsic
     * @see CameraDistortion
     */
    struct CameraModel {
        /// Camera intrinsics object
        CameraIntrinsic cI;
        /// Camera distortion object
        CameraDistortion cD;
        
        /// Specified camera Resolution
        cv::Size capSize;
        
        /// Optimized Rectification Map for X;
        cv::Mat m1;

        /// Optimized Rectification Map for Y;
        cv::Mat m2;

        /**
         * @brief Constructs a CameraModel within the given parameters
         * @param cI_ Camera intrinsic object
         * @param cD_ Camera distortion object
         * @param capSize_ Camera resolution
         */
        CameraModel(CameraIntrinsic cI_, CameraDistortion cD_, cv::Size capSize_) :
            cI(cI_), cD(cD_), capSize(capSize_) {};

        CameraModel(cv::Size capSize_, CameraIntrinsic cI_ = CameraIntrinsic(), CameraDistortion cD_ = CameraDistortion()) :
            cI(cI_), cD(cD_), capSize(capSize_) {};
    };

    /**
     * @brief Frame data object to be handled by a MonoTracker instance.
     * (\ref StringSLAM::Feature::MonoTracker).
     */
    struct Frame {
        /// ID of the frame
        int id;

        /// Image set by capture device
        cv::Mat frame;

        /// Extracted Keypoints from frame.
        std::vector<cv::KeyPoint> kp;

        /// Matches description of this frame compared to another
        cv::Mat desc;

        /// Real time location of when frame was captured 
        cv::Mat pose;

        /// Timestamp of when frame was captured.
        std::chrono::system_clock::time_point timestamp;
        
        /**
         * @brief Set timestamp of Frame
         */
        void setTimestamp() {
            timestamp = std::chrono::system_clock::now();
        }

        /**
         * @brief Copy Frame data into this Frame.
         * @param f Original Frame.
         */
        void copyFrom(const Frame &f) {
            id = f.id;
            kp = f.kp;
            timestamp = f.timestamp;
            frame = f.frame.clone();
            desc = f.desc.clone();
            pose = f.pose.clone();
        }

        /**
         * @brief Set Frame pose from t (translation) and R (rotation).
         * @param t Translation Matrix
         * @param R Rotation Matrix
         */
        void setPoseFromTR(const cv::Mat &t, const cv::Mat &R) {
            if (R.rows != 3 || R.cols != 3 || t.rows != 3 || t.cols != 1) return;
            pose = cv::Mat::eye(4, 4, R.type());
            R.copyTo(pose(cv::Range(0,3), cv::Range(0,3)));
            t.copyTo(pose(cv::Range(0,3), cv::Range(3,4)));
        }

        /**
         * @brief Get r & T from current pose of Frame.
         * @param t Translation Matrix
         * @param R Rotation Matrix
         * @return Frame's translation and rotation matrix.
         */
        void getTRFromPose(cv::Mat &t, cv::Mat &R) {
            if (pose.empty() ||  pose.rows != 4 || pose.cols != 4) return;
            R = pose(cv::Range(0,3), cv::Range(0,3)).clone();
            t = pose(cv::Range(0,3), cv::Range(3,4)).clone();
        }
    };

    /**
     * @brief Stores the Stereo Cameras distortion parameters
     * 
     * The distortion defines the Rectification between two MonoTrackers.
     * Also the R & T from both cameras.
     */
    struct StereoCameraDistortion {
        /// Rotation from left -> right camera
        cv::Mat R;

        /// Translation from left -> right camera
        cv::Mat T;

        /// Rectification rotation for left camera
        cv::Mat R1;

        /// Rectification rotation for right camera
        cv::Mat R2;

        /// Projection matrix for left camera
        cv::Mat P1;

        /// Projection matrix for right camera
        cv::Mat P2;

        /// Disparity-to-depth matrix
        cv::Mat Q;

        StereoCameraDistortion(cv::Mat R_, cv::Mat T_) : R(R_), T(T_) {}
        StereoCameraDistortion() = default;
    };

    /**
     * @brief A Frame type used to interface with StereoTracker
     */
    struct StereoFrame {
        /// ID of the frame
        int id;

        /// Image set by left capture device
        Frame frameLeft;

        /// Image set by right capture device
        Frame frameRight;
        
        /// Depth map of frame
        cv::Mat depthFrame;

        /// Extracted Keypoints from frame.
        std::vector<cv::KeyPoint> kp;

        /// Matches description of this frame compared to another
        cv::Mat desc;

        /// Real time location of when frame was captured 
        cv::Mat pose;

        /// Timestamp of when frame was captured.
        std::chrono::system_clock::time_point timestamp;

        /**
         * @brief Set timestamp of StereoFrame
         */
        void setTimestamp() {
            timestamp = std::chrono::system_clock::now();
        }

        /**
         * @brief Copy StereoFrame data into this StereoFrame.
         * @param f Original StereoFrame.
         */
        void copyFrom(const StereoFrame &f) {
            id = f.id;
            kp = f.kp;
            timestamp = f.timestamp;
            frameLeft.copyFrom(f.frameLeft);
            frameRight.copyFrom(f.frameRight);
            depthFrame = f.depthFrame.clone();
            desc = f.desc.clone();
            pose = f.pose.clone();
        }

        /**
         * @brief Set StereoFrame pose from t (translation) and R (rotation).
         * @param t Translation Matrix
         * @param R Rotation Matrix
         */
        void setPoseFromTR(const cv::Mat &t, const cv::Mat &R) {
            if (R.rows != 3 || R.cols != 3 || t.rows != 3 || t.cols != 1) return;
            pose = cv::Mat::eye(4, 4, R.type());
            R.copyTo(pose(cv::Range(0,3), cv::Range(0,3)));
            t.copyTo(pose(cv::Range(0,3), cv::Range(3,4)));
        }

        /**
         * @brief Get r & T from current pose of StereoFrame.
         * @param t Translation Matrix
         * @param R Rotation Matrix
         * @return StereoFrame's translation and rotation matrix.
         */
        void getTRFromPose(cv::Mat &t, cv::Mat &R) {
            if (pose.empty() ||  pose.rows != 4 || pose.cols != 4) return;
            R = pose(cv::Range(0,3), cv::Range(0,3)).clone();
            t = pose(cv::Range(0,3), cv::Range(3,4)).clone();
        }
    };

    // ------------------ Classes ------------------

    /**
     * @brief CV Wrapper for StereoSGBM
     */
    class StereoSGBMWrapper {
        private:
            cv::Ptr<cv::StereoSGBM> stereo;
            cv::Mat disp;
        public:
            StereoSGBMWrapper(
                int minDisparity=0, 
                int numDisparities=16, 
                int blockSize=3, 
                int P1=0, 
                int P2=0, 
                int disp12MaxDiff=0, 
                int preFilterCap=0, 
                int uniquenessRatio=0, 
                int speckleWindowSize=0, 
                int speckleRange=0, 
                int mode=cv::StereoSGBM::MODE_SGBM
            ) {
                stereo = cv::StereoSGBM::create(
                    minDisparity,
                    numDisparities,
                    blockSize,
                    P1,
                    P2,
                    disp12MaxDiff,
                    preFilterCap,
                    uniquenessRatio,
                    speckleWindowSize,
                    speckleRange,
                    mode
                );
            }

            ~StereoSGBMWrapper() = default;

            inline void compute(StereoFrame &sf) {
                stereo->compute(sf.frameLeft.frame, sf.frameRight.frame, disp);
                disp.convertTo(sf.depthFrame, CV_32F, 1.0 / 16.0);
            }

            inline static std::shared_ptr<StereoSGBMWrapper> create(
                int minDisparity=0, 
                int numDisparities=16, 
                int blockSize=3, 
                int P1=0, 
                int P2=0, 
                int disp12MaxDiff=0, 
                int preFilterCap=0, 
                int uniquenessRatio=0, 
                int speckleWindowSize=0, 
                int speckleRange=0, 
                int mode=cv::StereoSGBM::MODE_SGBM
            ) {
                return std::make_shared<StereoSGBMWrapper>(
                    minDisparity,
                    numDisparities,
                    blockSize,
                    P1,
                    P2,
                    disp12MaxDiff,
                    preFilterCap,
                    uniquenessRatio,
                    speckleWindowSize,
                    speckleRange,
                    mode
                );
            }
    };
    
    /**
     * @brief A wrapper class for OpenCV's ORB.
     */
    class OrbWrapper {

        private:
            cv::Ptr<cv::ORB> orb;
        public:
            /**
             * @brief Construct a OpenCV OrbWrapper
             * 
             * Same parameters as ORB::create(), refer to OpenCV docs.
             */
            OrbWrapper(            
                int nfeatures,
                float scaleFactor,
                int nlevels,
                int edgeThreshold,
                int firstLevel,
                int WTA_K,
                cv::ORB::ScoreType scoreType,
                int patchSize,
                int fastThreshold
            ) {
                orb = cv::ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, scoreType, patchSize, fastThreshold);
            }

            ~OrbWrapper() = default;
            
            /**
             * @brief Get the max features specified from initialization
             * @return Max ORB Features
             */
            inline int getMaxFeatures() const { return orb->getMaxFeatures(); };

            /**
             * @brief Get fast threshold specified  from initialization
             * @return Fast Threshold Parameter
             */
            inline int getFastThreshold() const { return orb->getFastThreshold(); };

            /**
             * @brief Refer to OpenCV doc.
             */
            inline void detectAndCompute(cv::Mat &img, std::vector<cv::KeyPoint> &kp, cv::Mat &desc) { 
                orb->detectAndCompute(img, cv::noArray(), kp, desc); 
            };

            /**
            * @brief Create Shared Pointer of OrbWrapper object
            * @return Shared Pointer of OrbWrapper
            */
            static std::shared_ptr<OrbWrapper> create(
                int nfeatures_ = 2000,
                float scaleFactor_ = 1.2f,
                int nlevels_ = 4,
                int edgeThreshold_ = 30,
                int firstLevel_ = 0,
                int WTA_K_ = 2,
                cv::ORB::ScoreType scoreType_ = cv::ORB::HARRIS_SCORE,
                int patchSize_ = 30,
                int fastThreshold_ = 50
            ) { 
                return std::make_shared<OrbWrapper>(nfeatures_, scaleFactor_, nlevels_, edgeThreshold_, firstLevel_, WTA_K_, scoreType_, patchSize_, fastThreshold_);
            };

    };

    /**
     * @brief A helper class to increase performance.
     * 
     * Find and enables OpenCV GPU Acceleration.
     */
    class Accelerator
    {
    private:
        cv::ocl::Context context;   // Device Context
        cv::ocl::Device device;     // Actual Device

    public:
        /**
         * Constructs Accelerator
         */
        Accelerator() = default;
        ~Accelerator() = default;

        /**
         * @brief Create GPU context
         * 
         * Create GPU Context, enable its usage, and the usage of threads.
         * @param useGPU Enable GPU Acceleration
         * @param threads Specify thread amounts to be used
         */
        void createGPU(bool useGPU = true, int threads = 0) {
            // Set Optimizations to true or false
            cv::setNumThreads(threads);
            cv::setUseOptimized(useGPU);
            //cv::setUseOpenVX(useGPU);

            // Create context, if it fails return
            if (!context.create(cv::ocl::Device::TYPE_GPU)) return;

            // Set device to context and enable OpenCL
            device = context.device(0);
            cv::ocl::setUseOpenCL(useGPU);
        };

        /**
         * @brief Get created GPU Context
         * @return GPU Context
         */
        inline cv::ocl::Context getContext() const { return context; }

        /**
         * @brief Get created GPU device
         * @return GPU Device
         */
        inline cv::ocl::Device getDevice() const { return device; }

        /**
         * @brief Create Shared Pointer of Accelerator object
         * @return Shared Pointer of Accelerator
         */
        static inline std::shared_ptr<Accelerator> create() {
            return std::make_shared<Accelerator>();
        };
    };
} // namespace StringSLAM
