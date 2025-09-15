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
    try {
        std::shared_ptr<Accelerator> accelerator = Accelerator::create();
        // If GPU is optional, skip creating it for now.
        // accelerator->createGPU(true, 10);
        std::cout << "[INFO] Accelerator created successfully\n";
    } catch (const std::exception &e) {
        std::cerr << "[WARN] Accelerator init threw: " << e.what() << "\n";
    } catch (...) {
        std::cerr << "[WARN] Accelerator init threw unknown\n";
    }

    // -------------------------------
    // 1. Configure the camera 
    // -------------------------------
    CameraModel cm(
        CameraIntrinsic(200.0, 200.0, 320.0, 240.0), // fx, fy, cx, cy (cx/cy centered for 640x480)
        CameraDistortion(0.0, 0.0, 0.0, 0.0, 0.0),  // start with zero distortion to be safe
        Size(640, 480)                              // image size
    );

    // -------------------------------
    // 2. Create a MonoTracker for capturing frames
    // -------------------------------
    std::shared_ptr<Tracker::MonoTracker> mt1 = Tracker::MonoTracker::create(0, cm);
    if (!mt1) {
        std::cerr << "[FATAL] Failed to create MonoTracker\n";
        return 1;
    }
    if (!mt1->open()) {
        std::cerr << "[WARN] mt1->open() returned false; camera may not be available\n";
    } else {
        std::cout << "[INFO] Camera opened\n";
    }

    // -------------------------------
    // 3. Initialize ORB/FeatureFinder (kept commented if not used)
    // -------------------------------
    std::shared_ptr<OrbWrapper> orb = OrbWrapper::create(800, 1.2f, 4, 30, 0, 2, cv::ORB::HARRIS_SCORE, 32, 30); 
    std::shared_ptr<Feature::FeatureFinder> featureFinder = Feature::FeatureFinder::create(orb);

    std::vector<cv::DMatch> matches;
    std::deque<Frame> frames;
    Frame tempFrame, prevFrame;
    Mat output;

    for (;;) {
        // Read frame from camera
        mt1->read(tempFrame);

        // Guard: did we actually get pixels?
        if (tempFrame.frame.empty()) {
            std::cerr << "[WARN] Empty frame grabbed. Skipping iteration.\n";
            // small sleep to avoid a hot loop when camera fails
            cv::waitKey(10);
            continue;
        }

        // If the frame list isn't empty then get prev frame.
        if (!frames.empty()) prevFrame = frames.back();

        output = tempFrame.frame.clone(); // clone to avoid aliasing issues

        // Safety checks before any feature operations
        if (featureFinder) featureFinder->getKeypoints(tempFrame);

        if (!prevFrame.kp.empty() && !tempFrame.kp.empty()) {
            matches = featureFinder->matchFramesLK(prevFrame, tempFrame);
        }

        if (!matches.empty() && !prevFrame.kp.empty()) {
            featureFinder->drawMatches(prevFrame, tempFrame, matches, output);
        }

        // show only if not empty
        if (!output.empty()) {
            cv::imshow("Viewer", output);
        }

        int k = cv::waitKey(1);
        if (k == 27) break; // Esc to quit

        // Only add frame if it has keypoints (or at least image data)
        if (!tempFrame.kp.empty())
            frames.push_back(tempFrame);
        else
            ; // nothing

        // If more than 45 frames remove the first one
        if (frames.size() > 45)
            frames.pop_front();
    }

    return 0;
}
```

See examples folder for more info

git add .
git commit -m "