/**
    disparity_sliding_window_test.cpp
    Purpose: Test Disparity Sliding Window on pedestrian detection

    @author Julian Mueller, University of Ulm
    @version 1.1 02/03/18
*/


#include <disparity_sliding_window.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include <fstream>
#include <iomanip> // setprecision

//////////////// ORIGINAL CODE FROM KITTI /////////////////////////
int getdir (std::string dir, std::vector<std::string> &files)
{

    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(dir.c_str())) == NULL) {
        std::cout << "Error(" << errno << ") opening " << dir << std::endl;
        return errno;
    }
    while ((dirp = readdir(dp)) != NULL) {
        if (std::string(dirp->d_name).find(".png") != std::string::npos) {
            files.push_back(std::string(dirp->d_name));
        }
        }
    closedir(dp);
    return 0;
}

// holding bounding boxes for ground truth and detections
struct tBox {
  std::string  type;     // object type as car, pedestrian or cyclist,...
  double   x1;      // left corner
  double   y1;      // top corner
  double   x2;      // right corner
  double   y2;      // bottom corner
  double   alpha;   // image orientation
  tBox (std::string type, double x1,double y1,double x2,double y2,double alpha) :
    type(type),x1(x1),y1(y1),x2(x2),y2(y2),alpha(alpha) {}
};

// holding ground truth data
struct tGroundtruth {
  tBox    box;        // object type, box, orientation
  double  truncation; // truncation 0..1
  int32_t occlusion;  // occlusion 0,1,2 (non, partly, fully)
  double ry;
  double  t1, t2, t3;
  double h, w, l;
  tGroundtruth () :
    box(tBox("invalild",-1,-1,-1,-1,-10)),truncation(-1),occlusion(-1) {}
  tGroundtruth (tBox box,double truncation,int32_t occlusion) :
    box(box),truncation(truncation),occlusion(occlusion) {}
  tGroundtruth (std::string type,double x1,double y1,double x2,double y2,double alpha,double truncation,int32_t occlusion) :
    box(tBox(type,x1,y1,x2,y2,alpha)),truncation(truncation),occlusion(occlusion) {}
};

std::vector<tGroundtruth> loadGroundtruth(std::string file_name,bool &success) {

  // holds all ground truth (ignored ground truth is indicated by an index vector
  std::vector<tGroundtruth> groundtruth;
  FILE *fp = fopen(file_name.c_str(),"r");
  if (!fp) {
    success = false;
    return groundtruth;
  }
  while (!feof(fp)) {
    tGroundtruth g;
    char str[255];
    if (fscanf(fp, "%s %lf %d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                   str, &g.truncation, &g.occlusion, &g.box.alpha,
                   &g.box.x1,   &g.box.y1,     &g.box.x2,    &g.box.y2,
                   &g.h,      &g.w,        &g.l,       &g.t1,
                   &g.t2,      &g.t3,        &g.ry )==15) {
      g.box.type = str;
      groundtruth.push_back(g);
    }
  }
  fclose(fp);
  success = true;
  return groundtruth;
}

//////////////// END ORIGINAL CODE FROM KITTI /////////////////////////

void readCalib(std::string &file_name, cv::Mat &p_left, cv::Mat &p_right, cv::Mat &ro_rect) {

    // Load calib file
    std::ifstream infile(file_name);
    std::string line;

    // We are interested in left and right projection matrix and rectification matrix
    p_left = cv::Mat (3, 4, CV_32FC1);
    p_right = cv::Mat (3, 4, CV_32FC1);
    ro_rect = cv::Mat (3, 3, CV_32FC1);

    // Read all lines - one line = one matrix
    while (std::getline(infile, line)) {
        std::stringstream ss(line);
        std::string token;

        int i = 0;
        std::string first_elem;
        std::vector<float> values;

        // Separate white space delimited line
        while (std::getline(ss, token, ' ')) {

            // Check which matrix line contains
            if(i == 0) {
                first_elem = token;
                ++i;
                continue;
            }
            else {
                values.push_back(std::stof(token));
                ++i;
            }
        }

        // Extract left projection matrix
        if (first_elem == "P2:") {
            for(int i = 0; i < p_left.rows * p_left.cols; ++i) {
                p_left.at<float>(i) = values[i];
            }

        }

        // Extract right projection matrix
        if (first_elem == "P3:") {
            for(int i = 0; i < p_right.rows*p_right.cols; ++i) {
                p_right.at<float>(i) = values[i];
            }

        }

        // Extract rectification matrix
        if (first_elem == "R0_rect:") {
            for(int i = 0; i < ro_rect.rows*ro_rect.cols; ++i) {
                ro_rect.at<float>(i)=values[i];
            }

        }
    }

}

float intersectionOverUnion(const int &labelX, const int &labelY, const int &labelW, const int &labelH,
                                                 const int &hypX, const int &hypY, const int &hypW, const int &hypH) {

    // TODO: reimplement this if you want to be opencv independend
    cv::Rect l(labelX, labelY, labelW, labelH);
    cv::Rect h(hypX, hypY, hypW, hypH);
    float i = (l & h).area();
    float u = l.area() + h.area() - i;
    return (i / u);
}

int main(int argc, char** argv) {

    // Download the KITTI object detection dataset (http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d) and modify path
    std::string left_img_dir = "/scratch/fs2/KITTI/data_object_image_2/training/image_2/";
    std::string right_img_dir = "/scratch/fs2/KITTI/data_object_image_3/training/image_3/";
    std::string calib_dir = "/scratch/fs2/KITTI/data_object_calib/training/calib/";
    std::string label_dir = "/scratch/fs2/KITTI/training/label_2/";

    std::vector<std::string> files_left;

    getdir(left_img_dir,files_left);

    cv::Mat img_left, img_right, disp, disp_viz;

    // sort files by number
    std::sort(files_left.begin(), files_left.end());
    // Parameters for SGM
    int min_disparity = 2 ;
    int max_disparity = 114;
    int num_disparities =  max_disparity- min_disparity ;
    int window_size = 5 ;
    int p1 = 8 * 3 * (window_size * window_size);
    int p2= 32 * 3 * (window_size * window_size);
    int disp_max_diff = 1;
    int prefilter_cap = 0;
    int uniqueness_ratio = 10;
    int speckle_window_size = 100;
    int speckle_range = 32;
    bool full_dp = false;

    // Parameters for DSW
    float world_width = 0.6;
    float world_height = 1.73;
    float aspect_ratio = 2.88;
    int min_width_image = 10;
    int max_width_image = 200;
    int class_id = 10;
    size_t nan_count = 4;
    float stddev = 1.0;
    float step_perc = 0.3;
    int class_type = DisparitySlidingWindow::HOMOGENEITY_VERIFICATION::PEDESTRIAN;


    // Constructors for SGM and Disparity Sliding Window
    cv::StereoSGBM SGM(min_disparity, num_disparities, window_size, p1, p2, disp_max_diff, prefilter_cap, uniqueness_ratio, speckle_window_size, speckle_range, full_dp);
    DisparitySlidingWindow DSW(world_width, world_height, aspect_ratio, min_width_image, max_width_image, class_id, nan_count, stddev, step_perc, class_type);
    // For all images
    for (size_t i = 0; i < files_left.size(); ++i){

        size_t lastindex = files_left[i].find_last_of(".");
        std::string rawname = files_left[i].substr(0, lastindex);

        // Get corresponding right image, calib and label files
        std::string right_img_path = right_img_dir + rawname + ".png";
        std::string calib_path = calib_dir + rawname + ".txt";
        std::string label_path = label_dir + rawname + ".txt";
        std::string left_img_path = left_img_dir + files_left[i];

        // Load Ground Truths from File
        bool success=false;
        std::vector<tGroundtruth> gts = loadGroundtruth(label_path, success);

        bool contains_ped = false;

        // Check if image contains pedestrian
        for (size_t i = 0; i < gts.size(); ++i) {

            if (gts[i].box.type == "Pedestrian") {

                contains_ped = true;
                break;
            }
        }

        // Continue if image contains pedestrian
        if (contains_ped) {

            img_left = cv::imread(left_img_path, -1);
            img_right = cv::imread(right_img_path, -1);

            cv::Mat p_left, p_right, ro_rect;
            readCalib(calib_path, p_left, p_right, ro_rect);

            cv::Mat K, R, t, dist;
            dist = cv::Mat (1, 5, CV_32FC1, cv::Scalar(0.));

            // Get camera matrix from projection matrix
            cv::decomposeProjectionMatrix(p_left, K, R, t);

            // Calculate SGM disparity image
            SGM.operator ()(img_left, img_right, disp);
            disp.convertTo(disp, CV_32F, 1.0/16.0, 0.0);

            // 1 means invalid
            disp.setTo(std::numeric_limits<float>::quiet_NaN (), disp ==1.0);

            float tx =  p_right.at<float>(0,3);

            // Calculate lookup-table (necessary in KITTI because calibration changes, for constant calib only calculate once!)
            DSW.initLookUpTable(tx, K, dist, 0, 114, 1./16. );
            cv::Mat dst;
            std::vector<Rect> hyps;
            clock_t begin = clock();
            DSW.generate(disp, dst, hyps, tx);
            clock_t end = clock();
            double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

            std::cout << "Elapsed miliseconds: " << elapsed_secs * 1000.0 << std::endl;
            // Calculate overlap for all ground truths
            for (size_t i = 0; i < gts.size(); ++i) {

                // Only do pedestrians
                if (gts[i].box.type == "Pedestrian") {
                    cv::rectangle(img_left, cv::Rect(gts[i].box.x1, gts[i].box.y1, gts[i].box.x2 - gts[i].box.x1, gts[i].box.y2 -gts[i].box.y1), cv::Scalar(0, 255, 0), 2);

                    float best_ov = 0.0;
                    int idx_best = 0;

                    // For all hypotheses calculate IoU and remember the best
                    for (size_t j = 0 ;j < hyps.size() ;++j) {
                        float iou = intersectionOverUnion(gts[i].box.x1, gts[i].box.y1, gts[i].box.x2 - gts[i].box.x1, gts[i].box.y2 - gts[i].box.y1, hyps[j].x_, hyps[j].y_, hyps[j].w_, hyps[j].h_);
                        if ( iou > best_ov) {
                            best_ov = iou;
                            idx_best = j;
                        }

                    }
                    // Visualize
                    if (best_ov > 0.0) {
                        std::stringstream stream;
                        stream << std::fixed << std::setprecision(2) << best_ov;
                        std::string s = stream.str();
                        cv::rectangle(img_left, cv::Rect(hyps[idx_best].x_, hyps[idx_best].y_, hyps[idx_best].w_, hyps[idx_best].h_), cv::Scalar(255, 255, 255),2);
                        cv::putText(img_left, "Best IoU: " + s, cv::Point2d(hyps[idx_best].x_  ,hyps[idx_best].y_ -5 ), CV_FONT_HERSHEY_PLAIN, 1,  cv::Scalar(0,255,0));
                    }
                }
            }

            // Visualize
            cv::putText(img_left, "Number of Proposals: " + std::to_string(hyps.size()), cv::Point2d(20, 350 ), CV_FONT_HERSHEY_PLAIN, 2,  cv::Scalar(0,255,0));
            cv::imshow("", img_left);
            cv::waitKey(0);
        }

    }

    return 0;

}
