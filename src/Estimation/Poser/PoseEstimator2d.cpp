#include <StringSLAM/Estimation/Poser/PoseEstimator2d.hpp>
#include <eigen3/Eigen/src/Core/Matrix.h>

namespace StringSLAM::Estimation::Poser
{
    Eigen::Vector2d PoseEstimator2d::transform(const Pose2D &pose, const Eigen::Vector2d &p, double c, double s) {
        return Eigen::Vector2d(c*p.x() - s*p.y() + pose.pos.x(),
                               s*p.x() + c*p.y() + pose.pos.y());
    }

    Eigen::Vector2d PoseEstimator2d::residual(const Pose2D &pose, const Eigen::Vector2d &p, const Eigen::Vector2d &q, double c, double s) {
        return transform(pose, p, c, s) - q;
    }

    Eigen::Matrix<double,2,3> PoseEstimator2d::jacobian_pose(double c, double s, const Eigen::Vector2d &p) {
        Eigen::Matrix<double,2,3> J;
        J << 1.0, 0.0, -s*p.x() - c*p.y(),
             0.0, 1.0,  c*p.x() - s*p.y();
        return J;
    }

    bool PoseEstimator2d::solvePose2D_GN(
        const std::vector<Eigen::Vector2d> &pts_p, const std::vector<Eigen::Vector2d> &pts_q,
        Pose2D &pose, int max_iters, double tol, bool useLM, double init_damping
    ) {
        if (pts_p.size() != pts_q.size() || pts_p.empty()) return false;
        const size_t N = pts_p.size();
        double lambda = init_damping;

        Eigen::Vector3d dx;

        for (int iter = 0; iter < max_iters; ++iter) {
            // Precompute trig
            double theta = pose.pos.z();
            double c = std::cos(theta);
            double s = std::sin(theta);

            Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
            Eigen::Vector3d b = Eigen::Vector3d::Zero();
            double total_error = 0.0;

            for (size_t i = 0; i < N; ++i) {
                const Eigen::Vector2d &p_i = pts_p[i];
                const Eigen::Vector2d &q_i = pts_q[i];

                // Inline residual instead of separate function
                Eigen::Vector2d r(
                    c*p_i.x() - s*p_i.y() + pose.pos.x() - q_i.x(),
                    s*p_i.x() + c*p_i.y() + pose.pos.y() - q_i.y()
                );
                total_error += r.squaredNorm();

                // Inline Jacobian to avoid returning a temporary
                Eigen::Matrix<double,2,3> J;
                J(0,0) = 1.0; J(0,1) = 0.0; J(0,2) = -s*p_i.x() - c*p_i.y();
                J(1,0) = 0.0; J(1,1) = 1.0; J(1,2) =  c*p_i.x() - s*p_i.y();

                H.noalias() += J.transpose() * J;
                b.noalias() += J.transpose() * r;
            }

            if (useLM) H.diagonal().array() += lambda;

            Eigen::LLT<Eigen::Matrix3d> llt(H);
            if (llt.info() != Eigen::Success) {
                lambda *= 10.0;
                if (lambda > 1e8) return false;
                continue;
            }

            dx = -llt.solve(b);

            Pose2D new_pose = pose;
            new_pose.pos.x() += dx(0);
            new_pose.pos.y() += dx(1);
            new_pose.pos.z() += dx(2);

            if (useLM) {
                // Recompute trig for new pose (was missing before!)
                double nc = std::cos(new_pose.pos.z());
                double ns = std::sin(new_pose.pos.z());

                double new_error = 0.0;
                for (size_t i = 0; i < N; ++i) {
                    const Eigen::Vector2d &p_i = pts_p[i];
                    const Eigen::Vector2d &q_i = pts_q[i];

                    double x = nc*p_i.x() - ns*p_i.y() + new_pose.pos.x();
                    double y = ns*p_i.x() + nc*p_i.y() + new_pose.pos.y();
                    Eigen::Vector2d r(x - q_i.x(), y - q_i.y());
                    new_error += r.squaredNorm();
                }

                if (new_error < total_error) {
                    pose = new_pose;
                    lambda = std::max(1e-9, lambda * 0.1);
                } else {
                    lambda *= 10.0;
                }
            } else {
                pose = new_pose;
            }

            if (dx.norm() < tol) return true;
        }
        return true;
    }

} // namespace StringSLAM::Estimation::Poser
