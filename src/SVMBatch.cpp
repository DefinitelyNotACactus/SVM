//
//  SVMBatch.cpp
//  SVM
//
//  Created by David Galvao on 19/09/22.
//

#include "SVMBatch.hpp"
#include "SVMParameters.hpp"
#include "Util.hpp"

#include <unordered_set>
#include <iostream>

SVMBatch::SVMBatch(svm_parameter *params, double pct, double epsilon) : AbstractSVM(params), pct(pct), epsilon(epsilon) { }

void SVMBatch::fit(const svm_problem &problem) {
    // Create S
    svm_problem S(problem.l, 0, problem.d);
    std::vector<int> initialSample = randomChoice(problem.l, (int) ceil(pct * (double) problem.l));
    for(int sample : initialSample) {
        S.append(problem.x[sample], problem.y[sample]);
    }
    // Create U
    svm_problem U(problem.l - S.l, problem.d), Ut(problem.l - S.l, problem.d);
    std::unordered_set<int> sampleSet(initialSample.begin(), initialSample.end());
    for(int row = 0, backIndex = 0; row < problem.l; row++) {
        if(sampleSet.find(row) != sampleSet.end()) {
            continue;
        }
        U.y[backIndex] = problem.y[row];
        Ut.y[backIndex] = problem.y[row];
        U.x[backIndex] = problem.x[row];
        Ut.x[backIndex] = problem.x[row];
        backIndex++;
    }
    // Main loop
    svm_model *currentModel;
    double bestScore = INFINITY;
    for(int itr = 0; itr < 10; itr++) {
        std::cout << "<Itr " << itr << "> len(U') = " << Ut.l << " len(S) = " << S.l << "\n";
        currentModel = svm_train(&S, params);
        SVMParameters modelParams = SVMParameters(currentModel, S, true);
        double *predS = predict(currentModel, S), *predU = predict(currentModel, U);
        double scoreS = accuracyScore(S.l, predS, S.y), scoreU = accuracyScore(U.l, predU, U.y);
        std::cout << "Score(S): " << scoreS << " Score(U): " << scoreU << "\n";

        if(abs(scoreS - scoreU) < bestScore) {
            bestScore = abs(scoreS - scoreU);
            if(itr > 0) { svm_free_model_content(model); }
            model = currentModel;
            if(bestScore < epsilon) {
                std::cout << "(Break) abs(Score(S) - Score(U)) < epsilon\n";
                delete [] predS;
                delete [] predU;
                break;
            }
        }
        if(Ut.l == 0) {
            std::cout << "(Break) U' is empty\n";
            delete [] predS;
            delete [] predU;
            break;
        }
        std::unordered_set<int> SVIndices;
        // Get all SVs on U
        for(int i = 0; i < Ut.l; i++) {
            if(predU[i] != Ut.y[i]) {
                SVIndices.insert(i);
                S.append(Ut.x[i], Ut.y[i]);
                //std::cout << "New SV: UX[" << i << "] h(x) != f(x):" << U.y[i] << "\n";
            } else {
                double leftHand = modelParams.computeLeftHand(Ut.x[i], Ut.y[i]);
                //std::cout << "<LH> UX[" << i << "]: " << leftHand << "\n";
                if (leftHand <= 1.0) {
                    SVIndices.insert(i);
                    S.append(Ut.x[i], Ut.y[i]);
                    //std::cout << "New SV: UX[" << i << "]" << leftHand << "\n";
                }
            }
        }
        delete [] predS;
        delete [] predU;
        
        if(SVIndices.empty()) {
            std::cout << "(Break) No SV in U\n";
            break;
        }
        // Remove all SVs from U
        Ut.l -= SVIndices.size();
        if(Ut.l > 0) {
            for(int i = 0, backIndex = 0; i < Ut.l + SVIndices.size(); i++) {
                if(SVIndices.find(i) == SVIndices.end()) { // UX[i] is not a SV
                    Ut.x[backIndex] = Ut.x[i];
                    Ut.y[backIndex] = Ut.y[i];
                    backIndex++;
                }
            }
        }
    }
    // Separate U by the labels
//    std::tuple<svm_node **, int, svm_node **, int> classes = separate_classes(problem);
    // Sample elements from both classes
//    std::vector<int> sample0 = randomChoice(std::get<1>(classes), ceil(pct * (double) std::get<1>(classes)));
//    std::vector<int> sample1 = randomChoice(std::get<3>(classes), ceil(pct * (double) std::get<3>(classes)));
//    for(int i = 0, backIndex = 0, lenClass = std::get<1>(classes); i < 2; i++) {
//        for(int j = 0; j < lenClass; j++) {
//            SX[backIndex] = new svm_node[problem.d + 1];
//            for(int k = 0; k < problem.d; k++) {
//                SX[backIndex][k] = std::get<0>(classes)[j][k];
//                Sy[backIndex] = -1;
//            }
//            SX[backIndex][problem.d] = svm_node();
//            backIndex++;
//        }
//        lenClass = std::get<3>(classes);
//    }
}

std::tuple<svm_node **, int, svm_node **, int> SVMBatch::separate_classes(svm_problem &problem) {
    svm_node **X0 = new svm_node*[problem.l], **X1 = new svm_node*[problem.l];
    int backIndex0 = 0, backIndex1 = 0;
    for(int i = 0; i < problem.l; i++) {
        if(problem.y[i] == 0) {
            X0[backIndex0] = new svm_node[problem.d + 1];
        } else {
            X1[backIndex1] = new svm_node[problem.d + 1];
        }
        for(int j = 0; j < problem.d; j++) {
            svm_node node;
            node.index = problem.x[i][j].index;
            node.value = problem.x[i][j].value;
            if(problem.y[i] == 0) {
                X0[backIndex0][j] = node;
            } else {
                X1[backIndex1][j] = node;
            }
        }
        svm_node endNode;
        endNode.index = -1;
        if(problem.y[i] == 0) {
            X0[backIndex0][problem.d + 1] = endNode;
            backIndex0++;
        } else {
            X1[backIndex1][problem.d + 1] = endNode;
            backIndex1++;
        }
    }
    return std::make_tuple(X0, backIndex0, X1, backIndex1);
}
