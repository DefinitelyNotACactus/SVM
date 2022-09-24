//
//  CBD.cpp
//  SVM
//
//  Created by David Galvao on 24/09/22.
//

#include "CBD.hpp"
#include "kdtree.hpp"

#include <math.h>

CBD::CBD(svm_parameter *params, int K, double pctS) : AbstractSVM(params), K(K), pctS(pctS) { };

void CBD::fit(const svm_problem &problem) {
    CBDScore *scores = determineScores(K, problem);
    std::sort(scores, scores + problem.l, std::greater<CBDScore>());
    svm_problem S((int) round((double) problem.l * pctS), 0, problem.d);
    for(int i = 0; i < S.maxl; i++) {
        S.append(problem.x[scores[i].index], problem.y[scores[i].index]);
    }
    
    model = svm_train(&S, params);
    
    delete [] scores;
}

CBDScore * CBD::determineScores(int K, const svm_problem &problem) {
    if (K >= problem.l) K = problem.l - 1;
    CBDScore *scores = new CBDScore[problem.l];
    std::tuple<double **, int **> neighbors = computeNeighbors(K, problem);
    std::vector<double> t(problem.l, 0);
    double **distance = std::get<0>(neighbors), gamma = 0;
    int **indices = std::get<1>(neighbors), counter = 0;
    // Determine gamma
    for(int i = 0; i < problem.l; i++) {
        bool foundNearestOppositeNeighbor = false;
        scores[i] = CBDScore(i, 0);
        for(int j = 1; j <= K; j++) {
            if (problem.y[i] != problem.y[indices[i][j]]) {
                if(!foundNearestOppositeNeighbor) {
                    foundNearestOppositeNeighbor = true;
                    t[i] = distance[i][j];
                }
                gamma += distance[i][j] - t[i];
                counter++;
            }
        }
    }
    gamma /= (double) counter;
    // Determine instance score
    std::vector<int> contributions(problem.l, 0);
    for(int i = 0; i < problem.l; i++) {
        for(int j = 1; j <= K; j++) {
            if (problem.y[i] != problem.y[indices[i][j]]) {
                scores[indices[i][j]].score += exp(-1.0 * ((distance[i][j] - t[i]) / gamma));
                contributions[indices[i][j]]++;
            }
        }
    }
    for(int i = 0; i < problem.l; i++) {
        if(contributions[i] > 0) { scores[i].score /= (double) contributions[i]; }
        delete [] distance[i];
        delete [] indices[i];
    }
    delete [] distance;
    delete [] indices;
    
    return scores;
}

std::tuple<double **, int **> CBD::computeNeighbors(const int K, const svm_problem &problem) {
    double **distances = new double*[problem.l];
    int **indices = new int*[problem.l];
    Kdtree::KdNodeVector nodes;
    for(int row = 0; row < problem.l; row++) {
        std::vector<double> point(problem.d);
        distances[row] = new double[K + 1];
        indices[row] = new int[K + 1];
        for(int column = 0; column < problem.d; column++) {
            point[column] = problem.x[row][column].value;
        }
        nodes.push_back(Kdtree::KdNode(point, problem.x[row][problem.d].value));
    }
    // Build tree
    Kdtree::KdTree tree(&nodes);
    // Search
    for(int row = 0; row < problem.l; row++) {
        Kdtree::KdNodeVector result;
        tree.k_nearest_neighbors(nodes[row].point, K + 1, &result);
        for(int neighbor = 0; neighbor <= K; neighbor++) {
            indices[row][neighbor] = result[neighbor].index;
            distances[row][neighbor] = computeDistance(nodes[row].point, result[neighbor].point);
        }
    }
    return std::make_tuple(distances, indices);
}

double CBD::computeDistance(const std::vector<double> &a, const std::vector<double> &b) {
    double dist = 0;
    for(int i = 0; i < a.size(); i++) {
        dist += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return sqrt(dist);
}
