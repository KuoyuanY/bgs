/*
This file is part of BGSLibrary.

BGSLibrary is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

BGSLibrary is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with BGSLibrary.  If not, see <http://www.gnu.org/licenses/>.
*/
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include "package_bgs/bgslibrary.h"

void codeList(std::string *);
IBGS* getBGS(std::string*, int);
void eachCode(std::string *, std::string, IBGS*);
void eachFrame(std::string, std::string, IBGS*);

int main(int argc, char **argv){
    std::cout << "Using OpenCV " << CV_MAJOR_VERSION << "." << CV_MINOR_VERSION << "." << CV_SUBMINOR_VERSION << std::endl;

    /* Background Subtraction Methods */
    std::string codes[100] = {};
    codeList(codes);
    std::string algos[32] = {};
    for(int i = 17; i < 32; i++){//each algorithm
        IBGS* bgs;
        bgs = getBGS(algos + i, i);
        std::cout << "\nStarted " << algos[i] << "\n" << std::endl;
        eachCode(codes, algos[i], bgs);
        std::cout << "\nFinished " << algos[i] << "\n" << std::endl;
        cvWaitKey(0);
        delete bgs;
    }

    return 0;
}

void eachCode(std::string * codes, std::string algo, IBGS* bgs){
    for(int j = 0; j < 100; j ++){//each dataset
        std::string code = *(codes + j);
        std::ofstream myfile;

        auto start = std::chrono::high_resolution_clock::now();//calculate how long algorithm takes
        eachFrame(code, algo, bgs);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        // myfile.open("../IMBS/algorithm_analysis/" + code + "/speed/" + algo + ".txt");
        // myfile << "Elapsed time is: "<< elapsed.count() << " seconds\n";
        // myfile.close();

        std::cout << "\nFinished "<< code << "\n"<< std::endl;
    }
}

void eachFrame(std::string code, std::string algo, IBGS* bgs){
    int counter = 1;
    int frameNumber = 1;
    int key = 0;
    while (key != 'q'){//each frame
        std::stringstream ss, rr;
        rr << counter;
        ss << frameNumber;
        std::string ground;
        if(counter % 2 == 0){
            ground = "foreground";
            frameNumber++;
        }else{
            ground = "background";
        }
        std::string fileName = "../IMBS/data_jpg_input/" + code + "/" + code + "_" + ss.str() + "_" + ground + ".jpg";
        std::cout << "reading " << fileName << std::endl;

        cv::Mat img_input = cv::imread(fileName, CV_LOAD_IMAGE_COLOR);

        if (img_input.empty())
        break;

        //cv::imshow("input", img_input);

        cv::Mat img_mask;
        cv::Mat img_bkgmodel;

        bgs->setShowOutput(false);
        bgs->process(img_input, img_mask, img_bkgmodel);

        // by default, "bgs->process(.)" automatically shows the foreground mask image
        // or set "bgs->setShowOutput(false)" to disable
        std::string savefile = "../IMBS/data_masks/" + code + "/"+ algo + "/x" + rr.str() + ".png";
        std::cout << "saving to " << savefile << std::endl;
        cv::imwrite(savefile, img_mask);

        //if(!img_mask.empty())
        //  cv::imshow("Foreground", img_mask);
        //  do something
        key = cvWaitKey(33);
        counter ++;
    }
}

IBGS* getBGS(std::string* element, int i){
    IBGS *bgs;
    switch(i){
        case 0:
        bgs = new FrameDifference;
        *element = "FrameDifference";
        break;
        case 1:
        bgs = new StaticFrameDifference;
        *element = "StaticFrameDifference";
        break;
        case 2:
        bgs = new WeightedMovingMean;
        *element = "WeightedMovingMean";
        break;
        case 3:
        bgs = new WeightedMovingVariance;
        *element = "WeightedMovingVariance";
        break;
        case 4:
        bgs = new MixtureOfGaussianV2;
        *element = "MixtureOfGaussianV2";
        break;
        case 5:
        bgs = new AdaptiveBackgroundLearning;
        *element = "AdaptiveBackgroundLearning";
        break;
        case 6:
        bgs = new AdaptiveSelectiveBackgroundLearning;
        *element = "AdaptiveSelectiveBackgroundLearning";
        break;
        case 7:
        bgs = new KNN;
        *element = "KNN";
        break;
        case 8:
        bgs = new DPAdaptiveMedian;
        *element = "DPAdaptiveMedian";
        break;
        case 9:
        bgs = new DPGrimsonGMM;
        *element = "DPGrimsonGMM";
        break;
        case 10:
        bgs = new LBSimpleGaussian;
        *element = "LBSimpleGaussian";
        break;
        case 11:
        bgs = new LBFuzzyGaussian;
        *element = "LBFuzzyGaussian";
        break;
        case 12:
        bgs = new LBMixtureOfGaussians;
        *element = "LBMixtureOfGaussians";
        break;
        case 13:
        bgs = new LBAdaptiveSOM;
        *element = "LBAdaptiveSOM";
        break;
        case 14:
        bgs = new LBFuzzyAdaptiveSOM;
        *element = "LBFuzzyAdaptiveSOM";
        break;
        case 15:
        bgs = new VuMeter;
        *element = "VuMeter";
        break;
        case 16:
        bgs = new KDE;
        *element = "KDE";
        break;
        case 17:
        bgs = new MultiCue;
        *element = "MultiCue";
        break;
        case 18:
        bgs = new SigmaDelta;
        *element = "SigmaDelta";
        break;
        case 19:
        bgs = new TwoPoints;
        *element = "TwoPoints";
        break;
        case 20:
        bgs = new ViBe;
        *element = "ViBe";
        break;
        case 21:
        bgs = new DPZivkovicAGMM;
        *element = "DPZivkovicAGMM";
        break;
        case 22:
        bgs = new FuzzyChoquetIntegral;
        *element = "FuzzyChoquetIntegral";
        break;
        case 23:
        bgs = new FuzzySugenoIntegral;
        *element = "FuzzySugenoIntegral";
        break;
        case 24:
        bgs = new T2FMRF_UV;
        *element = "T2FMRF_UV";
        break;
        case 25:
        bgs = new T2FMRF_UM;
        *element = "T2FMRF_UM";
        break;
        case 26:
        bgs = new T2FGMM_UV;
        *element = "T2FGMM_UV";
        break;
        case 27:
        bgs = new T2FGMM_UM;
        *element = "T2FGMM_UM";
        break;
        case 28:
        bgs = new DPEigenbackground;
        *element = "DPEigenbackground";
        break;
        case 29:
        bgs = new DPPratiMediod;
        *element = "DPPratiMediod";
        break;
        case 30:
        bgs = new DPWrenGA;
        *element = "DPWrenGA";
        break;
        case 31:
        bgs = new DPMean;
        *element = "DPMean";
        break;
        default:
        bgs = new CodeBook;
        break;
    }

    return bgs;
}

void codeList(std::string* codes){
    std::string temp[100] = {
        // "0FA",
        // "0J4",
        // "0S9",
        "1CU",
        "2SE",
        "2TV",
        "3AV",
        "3V9",
        "4AJ",
        "5EI",
        "06G",
        "6E9",
        "6P9",
        "7TI",
        "7W4",
        "8GJ",
        "8O1",
        "9VI",
        "39U",
        "44S",
        "55J",
        "67Y",
        "85B",
        "92P",
        "94G",
        "98H",
        "485",
        "AE0",
        "AIE",
        "B7R",
        "BJN",
        "C7S",
        "CRV",
        "D1A",
        "DE8",
        "DVC",
        "E5M",
        "ELV",
        "FCQ",
        "FV1",
        "G0E",
        "G5P",
        "GE6",
        "GQ3",
        "H8B",
        "HJC",
        "HU2",
        "HW2",
        "I1U",
        "I71",
        "IVF",
        "JI9",
        "JLA",
        "K0L",
        "K3X",
        "KK5",
        "KPS",
        "LLS",
        "LPA",
        "M6H",
        "MO7",
        "NBX",
        "NM4",
        "NM6",
        "NRJ",
        "NTP",
        "OIV",
        "OQQ",
        "OV0",
        "PBL",
        "PYH",
        "QFE",
        "QQ5",
        "R1K",
        "RKZ",
        "RTF",
        "RU6",
        "S4H",
        "SJN",
        "SZY",
        "T12",
        "T83",
        "TWS",
        "UBY",
        "VL7",
        "VXC",
        "W2I",
        "X16",
        "XF6",
        "XK7",
        "XKM",
        "XPT",
        "Y0Q",
        "Y33",
        "YFN",
        "YP9",
        "Z74",
        "ZCR",
        "ZJV",
        "ZVN"
    };
    for(int i = 0; i < 100; i++){
        *(codes + i) = temp[i];
    }
}
