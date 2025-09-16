#pragma once
#include "opencv2/core/types.hpp"
#include <eigen3/Eigen/Eigen>
#include <eigen3/Eigen/src/Core/Matrix.h>

namespace StringSLAM::Estimation::Poser
{
    /**
     * A data structure for storing the 2d pose.
     */
    struct Pose2D {
        /// @brief Vector for holding the: X, & Y coordinate, aswell as theta.
        Eigen::Vector3f pos;

        /**
         * @brief Return theta
         * 
         * Could use .Z(), but a getter will exist to make things more clear
         */
        double getTheta() {
            return pos.z();
        }
        
        /// @brief Return our position as a matrix
        Eigen::Matrix3d matrix() const {
            Eigen::Matrix3d T = Eigen::Matrix3d::Identity();
            double c = cos(pos.z());
            double s = sin(pos.z());

            T(0,0) = c; T(0,1) = -s; T(0,2) = pos.x();
            T(1,0) = s; T(1,1) =  c; T(1,2) = pos.y();
            return T;
        }
    };

    /**
     * @brief A class for estimating pose from 2 Frame 's.
     * 
     * A class for estimating the current pose from non-depth source.
     */
    class PoseEstimator2d
    {
        public:
        PoseEstimator2d() = default;
        ~PoseEstimator2d() = default;

        /**
         * @brief Transform a point p (2x1) by a matrix.
         * @param pose Pose
         * @param p Point
         */
        Eigen::Vector2d transform(const Pose2D &pose, const Eigen::Vector2d &p, double c, double s);

        /**
         * @brief Compute the residual
         * 
         * r = R(theta) * p + t - q
         * @param pose Pose
         * @param p Point
         */
        Eigen::Vector2d residual(const Pose2D &pose, const Eigen::Vector2d &p, const Eigen::Vector2d &q, double c, double s);

        /**
         * @brief Calculate Jacobian from pose.
         * @param pose Pose
         * @param p Point
         */
        Eigen::Matrix<double,2,3> jacobian_pose(double c, double s, const Eigen::Vector2d &p);

        /**
         * @brief Gauss-Newton (with optional LM damping). Points must be corresponded:
         * pts_p[i] (in source frame) -> pts_q[i] (in target/world frame)
         */
        bool solvePose2D_GN(const std::vector<Eigen::Vector2d> &pts_p, const std::vector<Eigen::Vector2d> &pts_q,
            Pose2D &pose, int max_iters = 50, double tol = 1e-6, 
            bool useLM = true, double init_damping = 1e-3
        );
        
        /**
         * @brief Create Shared Pointer of PoseEstimator2d object
         * @return Shared Pointer of PoseEstimator2d
         */
        static std::shared_ptr<PoseEstimator2d> create() {
            return std::make_shared<PoseEstimator2d>();
        }
    };   
} // namespace StringSLAM::Estimation::Poser
