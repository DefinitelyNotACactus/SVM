//
//  SVMBatch.cpp
//  SVM
//
//  Created by David Galvao on 19/09/22.
//

#include "SVMBatch.hpp"
#include "Util.hpp"

#include <iostream>

SVMBatch::SVMBatch(svm_parameter *params, double pct, double epsilon) : AbstractSVM(params), pct(pct), epsilon(epsilon) { }

void SVMBatch::fit(const svm_problem &problem) {
    // Separate X by the labels
    std::tuple<svm_problem, svm_problem> classes = separate_classes(problem);
    svm_problem U0 = std::get<0>(classes), U1 = std::get<1>(classes);
    // Sample elements from both classes
    std::vector<int> sample0 = randomChoice(U0.l, ceil(pct * (double) U0.l)), sample1 = randomChoice(U1.l, ceil(pct * (double) U1.l));
    // Create S and U
    svm_problem S(problem.l, 0, problem.d);
    for(int sample : sample0) {
        S.append(U0.x[sample], U0.y[sample]);
    }
    for(int sample : sample1) {
        S.append(U1.x[sample], U1.y[sample]);
    }
    // Remove from U all the selected samples
    std::unordered_set<int> sampleSet0(sample0.begin(), sample0.end()), sampleSet1(sample1.begin(), sample1.end());
    removeSamples(sampleSet0, U0);
    removeSamples(sampleSet1, U1);
    // Main loop
    double bestScore = INFINITY;
    for(int itr = 0; itr < 100; itr++) {
        std::cout << "<Itr " << itr << "> len(U0) = " << U0.l << " len(U1) = " << U1.l << " len(S) = " << S.l << "\n";
        svm_model *currentModel = svm_train(&S, params);
        SVMParameters currentModelParams = SVMParameters(currentModel, S, false);
        double *predS = predict(currentModel, S), *predU0 = predict(currentModel, U0), *predU1 = predict(currentModel, U1), *predX = predict(currentModel, problem);
        double scoreS = accuracyScore(S.l, predS, S.y), scoreU0 = accuracyScore(U0.l, predU0, U0.y), scoreU1 = accuracyScore(U1.l, predU1, U1.y), scoreX = accuracyScore(problem.l, predX, problem.y);
        //double itrScore = abs(scoreS - (scoreU0 * (double) U0.l + scoreU1 * (double) U1.l) / (double) (U0.l + U1.l));
        double itrScore = abs(scoreS - scoreX);
        std::cout << "Score(S): " << scoreS << " Obj: " << itrScore << "\n";
        if (itrScore < bestScore) {
            if(itr > 0) {
                svm_free_model_content(model);
            }
            bestScore = itrScore;
            model = currentModel;
            if(bestScore < epsilon) {
                std::cout << "(Break) abs(Score(S) - Score(U)) < epsilon\n";
                delete [] predS;
                delete [] predU0;
                delete [] predU1;
                delete [] predX;
                break;
            }
        }
        // Get S proportions
        double wrongS0 = 0, nsvS0 = 0, svS0 = 0, wrongS1 = 0, nsvS1 = 0, svS1 = 0;
        for(int i = 0; i < S.l; i++) {
            if(predS[i] != S.y[i]) {
                if (S.y[i] == 1) { wrongS1++; }
                else { wrongS0++; }
            } else if(currentModelParams.isSV(i)) {
                if (S.y[i] == 1) { svS1++; }
                else { svS1++; }
            } else if(S.y[i] == 1) { nsvS1++; }
            else { nsvS0++; }
        }
        wrongS0 /= (double) S.l;
        nsvS0 /= (double) S.l;
        svS0 /= (double) S.l;
        wrongS1 /= (double) S.l;
        nsvS1 /= (double) S.l;
        svS1 /= (double) S.l;
        // Get the candidates
        std::unordered_set<int> candidatesH0, candidatesSV0, candidatesNSV0, candidatesH1, candidatesSV1, candidatesNSV1;
        getCandidates(candidatesH0, candidatesSV0, candidatesNSV0, predU0, currentModelParams, U0);
        getCandidates(candidatesH1, candidatesSV1, candidatesNSV1, predU1, currentModelParams, U1);
        
        double wrongU0 = candidatesH0.size(), nsvU0 = candidatesNSV0.size(), svU0 = candidatesSV0.size(), wrongU1 = candidatesH1.size(), nsvU1 = candidatesNSV1.size(), svU1 = candidatesSV1.size(), Ul = U0.l + U1.l;
        wrongU0 /= Ul;
        nsvU0 /= Ul;
        svU0 /= Ul;
        wrongU1 /= Ul;
        nsvU1 /= Ul;
        svU1 /= Ul;
        // Select some candidates from each partition
        std::unordered_set<int> toRemove0, toRemove1;
        selectCandidates(candidatesH0, toRemove0, U0, S);
        selectCandidates(candidatesSV0, toRemove0, U0, S);
        selectCandidates(candidatesNSV0, toRemove0, U0, S);
        selectCandidates(candidatesH1, toRemove1, U1, S);
        selectCandidates(candidatesSV1, toRemove1, U1, S);
        selectCandidates(candidatesNSV1, toRemove1, U1, S);
        // Remove all selected samples from U
        removeSamples(toRemove0, U0);
        removeSamples(toRemove1, U1);
        // Free memory
        delete [] predS;
        delete [] predU0;
        delete [] predU1;
        delete [] predX;
        if(U0.l == 0 and U1.l == 0) {
            std::cout << "(Break) U is empty\n";
            break;
        }
    }
}

std::tuple<svm_problem, svm_problem> SVMBatch::separate_classes(const svm_problem &problem) {
    svm_problem X0(problem.l, 0, problem.d), X1(problem.l, 0, problem.d);
    for(int i = 0; i < problem.l; i++) {
        if(problem.y[i] == 1) {
            X1.append(problem.x[i], problem.y[i]);
        } else {
            X0.append(problem.x[i], problem.y[i]);
        }
    }
    return std::make_tuple(X0, X1);
}

void SVMBatch::removeSamples(const std::unordered_set<int> &indices, svm_problem &problem) {
    // Remove all SVs from U
    problem.l -= indices.size();
    if(problem.l > 0) {
        for(int i = 0, backIndex = 0; i < problem.l + indices.size(); i++) {
            if(indices.find(i) == indices.end()) { // UX[i] is not a SV
                problem.x[backIndex] = problem.x[i];
                problem.y[backIndex] = problem.y[i];
                backIndex++;
            }
        }
    }
}

void SVMBatch::getCandidates(std::unordered_set<int> &candidatesH, std::unordered_set<int> &candidatesSV, std::unordered_set<int> &candidatesNSV, const double *predictions, SVMParameters &params, const svm_problem &problem) {
    for(int i = 0; i < problem.l; i++) {
        if(predictions[i] == problem.y[i]) {
            double leftHand = params.computeLeftHand(problem.x[i], problem.y[i]);
            if (leftHand <= 1) { candidatesSV.insert(i); }
            else { candidatesNSV.insert(i); }
        } else { candidatesH.insert(i); }
    }
}

void SVMBatch::selectCandidates(const std::unordered_set<int> &candidates, std::unordered_set<int> &toRemove, const svm_problem &problem, svm_problem &S) {
    int sampleSize = round(ceil(pct * candidates.size()));
    if(candidates.size() > 0 and sampleSize > 0) {
        if(candidates.size() < sampleSize) {
            toRemove.insert(candidates.begin(), candidates.end());
            for(int sample : candidates) {
                S.append(problem.x[sample], problem.y[sample]);
                toRemove.insert(sample);
            }
        } else {
            std::vector<int> selected = randomChoice((int) candidates.size(), sampleSize);
            toRemove.insert(selected.begin(), selected.end());
            for(int sample : selected) {
                S.append(problem.x[sample], problem.y[sample]);
                toRemove.insert(sample);
            }
        }
    }
}
