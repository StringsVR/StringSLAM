#pragma once
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/ocl.hpp>

namespace StringSLAM
{
    // ---------- Structs -----------

    // Camera Data
    struct CameraModel {
        cv::Mat K;          // Camera Intrinsics
        cv::Mat D;          // Camera Distortion
        cv::Mat R;          // Still don't know what this does ngl
        cv::Size capSize;   // Camera Resolution (Or what you want it set as)

        CameraModel() = default;
        CameraModel(const cv::Mat &K_, cv::Mat &D_, cv::Mat &R_, cv::Size &capSize_)
            : K(K_.clone()), D(D_), R(R_), capSize(capSize_) {};
    };

    // Frame Data
    struct Frame
    {
        int id;                         // ID of frame.
        cv::Mat frame;                  // Actual Frame
        std::vector<cv::KeyPoint> kp;   // Keypoints of frame
        cv::Mat desc;                   // Descriptor of keypoints
        cv::Mat pose;                   // 4x4 pose: [R | t; 0 0 0 1]

        Frame() = default;

        void copyFrom(const Frame &other) {
            id = other.id;
            frame = other.frame.clone();
            kp = other.kp;
            desc = other.desc.clone();
            pose = other.pose.clone();
        }

        // Set pose from rotation R (3x3) and translation t (3x1)
        void setPoseFromRt(const cv::Mat &R, const cv::Mat &t) {
            CV_Assert(R.rows == 3 && R.cols == 3 && t.rows == 3 && t.cols == 1);
            pose = cv::Mat::eye(4, 4, R.type());
            R.copyTo(pose(cv::Range(0,3), cv::Range(0,3)));
            t.copyTo(pose(cv::Range(0,3), cv::Range(3,4)));
        }

        // Extract rotation R (3x3) and translation t (3x1) from pose
        void getRt(cv::Mat &R, cv::Mat &t) const {
            CV_Assert(!pose.empty() && pose.rows == 4 && pose.cols == 4);
            R = pose(cv::Range(0,3), cv::Range(0,3)).clone();
            t = pose(cv::Range(0,3), cv::Range(3,4)).clone();
        }
    };

    // ---------- Classes -----------

    // Wrapper for Ptr<ORB>, Instead of passing we have to make one bc OpenCV bug.
    // Standard ORB Wrapper, you can read the opencv docs if interested.
    class OrbWrapper
    {
    private:
        cv::Ptr<cv::ORB> orb;
    public:
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
        };

        ~OrbWrapper() = default;

        inline int getMaxFeatures() const { return orb->getMaxFeatures(); };
        inline int getFastThreshold() const { return orb->getFastThreshold(); };

        inline void detectAndCompute(cv::Mat &img, std::vector<cv::KeyPoint> &kp, cv::Mat &desc) { 
            orb->detectAndCompute(img, cv::noArray(), kp, desc); 
        };

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

    // Create A GPU/CPU Accelerator for tasks
    class Accelerator
    {
    private:
        cv::ocl::Context context;   // Device Context
        cv::ocl::Device device;     // Actual Device

    public:
        Accelerator() = default;
        ~Accelerator() = default;

        // Create GPU Context
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

        inline cv::ocl::Context getContext() const { return context; }
        inline cv::ocl::Device getDevice() const { return device; }

        static inline std::shared_ptr<Accelerator> create() {
            return std::make_shared<Accelerator>();
        };
    };

} // namespace StringSLAM   