#include <iostream>
#include <string>
#include <map>

#include "kernels/cublas-mix.cuh"
#include "kernels/wmma-mix.cuh"
#include "kernels/basic_tiling.cuh"

int main(int argc, char* argv[]) {
    // Variables to store flag values
    std::map<std::string, int> requiredFlags;
    bool bFlag = false;
    bool wFlag = false;
    bool lFlag = false;

    // Loop through arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        // Handle required flags with integer values
        if (arg.rfind("-m=", 0) == 0) {
            requiredFlags["m"] = std::stoi(arg.substr(3));
        } else if (arg.rfind("-n=", 0) == 0) {
            requiredFlags["n"] = std::stoi(arg.substr(3));
        } else if (arg.rfind("-k=", 0) == 0) {
            requiredFlags["k"] = std::stoi(arg.substr(3));
        }
        // Handle boolean flags
        else if (arg == "-b") {
            bFlag = true;
        } else if (arg == "-w") {
            wFlag = true;
        } else if (arg == "-l") {
            lFlag = true;
        } else {
            std::cerr << "Unknown flag: " << arg << std::endl;
            return 1;
        }
    }

    // Check if all required flags are present
    if (requiredFlags.find("m") == requiredFlags.end() ||
        requiredFlags.find("n") == requiredFlags.end() ||
        requiredFlags.find("k") == requiredFlags.end()) {
        std::cerr << "Required flags -m=, -n=, and -k= are missing" << std::endl;
        return 1;
    }

    // Output values for testing
    std::cout << "-m= " << requiredFlags["m"] << std::endl;
    std::cout << "-n= " << requiredFlags["n"] << std::endl;
    std::cout << "-k= " << requiredFlags["k"] << std::endl;
    std::cout << "-b run cublas" << (bFlag ? "true" : "false") << std::endl;
    std::cout << "-w run wmma" << (wFlag ? "true" : "false") << std::endl;
    std::cout << "-l run basic tiling" << (lFlag ? "true" : "false") << std::endl;

    int M = requiredFlags["m"];
    int N = requiredFlags["n"];
    int K = requiredFlags["k"];

    if (bFlag) {
	    std::cout << "cublas time (ms):" << run_cublas(M, N, K) << std::endl;
    }
    if (wFlag) {
	    std::cout << "wmma time (ms):" << run_wmma(M, N, K) << std::endl;
    }
    if (lFlag) {
	    std::cout << "basic tiling time (ms):" << run_basic_tiling(M, N, K) << std::endl;
    }

    std::cout << "done" << std::endl;
    return 0;
}
