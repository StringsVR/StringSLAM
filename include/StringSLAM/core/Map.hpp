#pragma once
#include "StringSLAM/core.hpp"
#include <map>
#

#include "opencv2/core/types.hpp"
namespace StringSLAM
{
    /**
     * @brief A data structure for a singular 3d point on a map.
     */
    struct MapPoint {
        /// @brief  Actual 3D Point
        cv::Point3f pos;

        /// @brief List of frame ID's that can see the initialized point.
        std::vector<int> observations;
    };

    /**
     * @brief A class meant to manage landmarks and keyframes.
     */
    class Map
    {
    private:
        std::map<int, Frame> keyframes;
        std::map<int, MapPoint> landmarks;
    public:
        /// @brief Create Constructor
        Map() = default;
        ~Map() = default;

        /**
         * @brief Add keyframe to Map
         * @param f Linked frame
         */
        inline void addKeyframe(const Frame& f) {
            keyframes[f.id] = f;
        }

        /**
         * @brief Add landmark to Map
         * @param mp Observed point
         */
        inline void addLandmark(const MapPoint& mp) {
            landmarks[landmarks.size()] = mp;
        }

        /**
         * @brief Get all keyframes.
         * @return Keyframes
         */
        const std::map<int, Frame>& getKeyframes() const { return keyframes; }

        /**
         * @brief Get all landmarks.
         * @return Landmarks
         */
        const std::map<int, MapPoint>& getLandmarks() const { return landmarks; }

        /**
         * @brief Create Shared Pointer of Map object
         * @return Shared Pointer of Map
         */
        static std::shared_ptr<Map> create() {
            return std::make_shared<Map>();
        }
    };
    
} // namespace StringSLAM
