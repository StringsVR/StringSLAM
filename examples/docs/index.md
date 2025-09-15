# Overview

## Introduction
StringSLAM is a lightweight and efficient SLAM (Simultaneous Localization and Mapping) library designed for the lower end hardware. 
The 2 main design philosphicies for this library are
- Low Power
- Customizable

It can provide fast feature tracking, frame matching, map generation for Robotics and AR/VR applications. \n
StringSLAM is easy to integrate and works with OpenCV.

### Why are there weird wrappers of OpenCV classes?
Ah yes. Well you see because calling Orb or a few other things while another library is using OpenCV seems to freeze your application. I'm frankly not sure if I suck, or the parallelization for this is just fried. Who knows 🤷, anywho, thats why you see dumb wrappers for OpenCV classes.

## Build
**Clone the repository and build with CMake:**

```
git clone https://github.com/StringsVR/StringSLAM.git
cd StringSLAM
mkdir build && cd build
cmake ..
```

**Generating documentation:**

```
cmake --build build --target docs
```

**Dependencies:**
```
- PkgConfig
- OpenCV
- Eigen
- DOxygen (If you want documentation)
```

## Quickstart ##
**Making a simple monocular matching system:**

```
int main() {
    // -------------------------------
    // 1. Configure the camera
    // -------------------------------
    CameraModel cm(
        CameraIntrinsic(200, 200, 200, 200),       // fx, fy, cx, cy
        CameraDistortion(15, 10, 12, 2, 5),       // k1, k2, p1, p2, k3
        Size(640, 480)                             // image size
    );

    // -------------------------------
    // 2. Create a MonoTracker for capturing frames
    // -------------------------------
    std::shared_ptr<Tracker::MonoTracker> tracker = Tracker::MonoTracker::create(0, cm);

    // -------------------------------
    // 3. Initialize ORB feature detector and FeatureFinder
    // -------------------------------
    std::shared_ptr<OrbWrapper> orb = OrbWrapper::create(
        800,                                // max number of features
        1.2f,                             // scale factor
        4,                                    // number of levels
        30, 0, 2,   // edge threshold, first level, WTA_K
        ORB::HARRIS_SCORE,              // score type
        32, 30              // patch size, fast threshold
    );

    std::shared_ptr<Feature::FeatureFinder> featureFinder = Feature::FeatureFinder::create(orb);

    // -------------------------------
    // 4. Prepare frame storage
    // -------------------------------
    std::deque<Frame> frames;
    Frame tempFrame, prevFrame;
    Mat output;

        // -------------------------------
    // 5. Main loop: read frames and track features
    // -------------------------------
    for (;;) {
        // Keep previous frame if available
        if (!frames.empty()) prevFrame = frames.back();

        // Capture new frame from camera
        tracker->read(tempFrame);
        output = tempFrame.frame;

        // Detect keypoints in current frame
        featureFinder->getKeypoints(tempFrame);

        // Match features with previous frame
        if (!prevFrame.kp.empty()) {
            auto matches = featureFinder->matchFramesLK(tempFrame, prevFrame);

            // Draw matches on the output image
            if (!matches.empty()) 
                featureFinder->drawMatch(tempFrame, prevFrame, matches, output);
        }

        // Display the frame
        imshow("Viewer", output);
        waitKey(tracker->getFPS());

        // Only store frames that have keypoints
        if (!tempFrame.kp.empty())
            frames.push_back(tempFrame);

        // Limit frame history to 45 frames
        if (frames.size() > 45)
            frames.pop_front();
    }

    return 0;
}
```

See examples folder for more info