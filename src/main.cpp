//
//  main.cpp
//  SVM
//
//  Created by David Galvao on 17/09/22.
//

#include <iostream>
#include <random>
#include <string>

#include "svm.hpp"
#include "DatasetReader.hpp"
#include "SVMParameters.hpp"

std::random_device random_device;
std::mt19937 generator(random_device());
std::uniform_real_distribution<double> realDistribution(0, 1);
std::uniform_int_distribution<int> intDistribution(0, 1);

svm_problem problem;
svm_parameter param;

int main(int argc, const char * argv[]) {
    DataFrame df = readCsv("datasets/heart-processed.csv");
    std::string target = "HeartDisease";
    // Params
    param.svm_type = C_SVC;
    param.kernel_type = RBF;
    param.degree = 3;
    param.gamma = (double) 1 / (double) (df.columns.size() - 2);    // 1/num_features
    param.coef0 = 0;
    param.nu = 0.5;
    param.cache_size = 100;
    param.C = 1;
    param.eps = 1e-3;
    param.p = 0.1;
    param.shrinking = 1;
    param.probability = 0;
    param.nr_weight = 0;
    param.weight_label = NULL;
    param.weight = NULL;
    // Problem
    problem.l = df.size;
    problem.d = (int) df.columns.size() - 2; // Ignore index and target
    problem.y = new double[problem.l];
    problem.x = new svm_node*[problem.l];
    for(int row = 0; row < problem.l; row++) {
        if(df[target][row] == 1) {
            problem.y[row] = 1;
        } else {
            problem.y[row] = -1;
        }
//        problem.y[row] = df[target][row];
        problem.x[row] = new svm_node[problem.d + 1];
        int backIndex = 0;
        for(int column = 0; column < df.columns.size(); column++) {
            if(df.columns[column] == "Index" or df.columns[column] == target) { continue; }
            svm_node node;
            node.index = column;
            node.value = df[df.columns[column]][row];
            problem.x[row][backIndex] = node;
            backIndex++;
        }
        svm_node endNode;
        endNode.index = -1;
        problem.x[row][backIndex] = endNode;
    }
    // Train
    svm_model *model = svm_train(&problem, &param);
    int wrongPredictions = 0;
    for(int i = 0; i < problem.l; i++) {
        double predict = svm_predict(model, problem.x[i]);
//        std::cout << "y=" << problem.y[i];
//        for(int j = 0; problem.x[i][j].index != -1; j++) {
//            std::cout << " (" << problem.x[i][j].index << "," << problem.x[i][j].value << ") ";
//        }
//        std::cout << "\n";
        if(predict != problem.y[i]) {
            std::cout << "x["<< i << "] h(x): " << predict << " f(x): " << problem.y[i] << "\n";
            wrongPredictions++;
        }
    }
    std::cout << "Accuracy: " << 1 - ((double) wrongPredictions / (double) df.size) << "\n";
    
    std::cout << "len(Dataset) = " << problem.l << "\n";
    std::cout << "nSV[0] = " << model->nSV[0] << " nSV[1] = " << model->nSV[1] << "\n";
    SVMParameters params(model, problem, true);
    
    delete model;
    
    return 0;
}
