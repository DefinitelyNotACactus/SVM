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
#include "SVMBatch.hpp"
#include "Util.hpp"

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
    problem = svm_problem(df.size, (int) df.columns.size() - 2);
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
            problem.x[row][backIndex] = svm_node(column, df[df.columns[column]][row]);
            backIndex++;
        }
        problem.x[row][backIndex] = svm_node(-1, row);
    }
    // Train
    svm_model *model = svm_train(&problem, &param);
    SVMParameters params(model, problem, true);
    std::cout << "len(Dataset) = " << problem.l << "\n";
    std::cout << "nSV[0] = " << model->nSV[0] << " nSV[1] = " << model->nSV[1] << "\n";
    SVMBatch batchModel(&param, 0.1, 0.01);
    batchModel.fit(problem);
    int wrongPredictionsBatch = 0, wrongPredictionsVanilla = 0;
    for(int i = 0; i < problem.l; i++) {
        double predictBatch = batchModel.predict(problem.x[i]), predictVanilla = batchModel.predict(model, problem.x[i]);
        if(predictBatch != problem.y[i]) {
//            std::cout << "X[" << i << "] h(X[i]) = " << predictBatch << " f(X[i]) = " << problem.y[i] << "\n";
            wrongPredictionsBatch++;
        }
        if(predictVanilla != problem.y[i]) {
            wrongPredictionsVanilla++;
        }
    }
    std::cout << "Vanilla:\n";
    classificationReport(problem.l, batchModel.predict(model, problem), problem.y);
    std::cout << "Batch:\n";
    classificationReport(problem.l, batchModel.predict(problem), problem.y);
    
    return 0;
}
