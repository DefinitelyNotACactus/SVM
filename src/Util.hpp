//
//  Util.hpp
//  SVM
//
//  Created by David Galvao on 18/09/22.
//

#ifndef Util_hpp
#define Util_hpp

#include "svm.hpp"

#include <vector>
#include <random>

extern std::random_device rd;
extern std::default_random_engine rng;

double dot(const int, const svm_node *, const svm_node *);
double dot(const int, const double *, const double *);
double norm(const int, const double *);
double *diff(const int, const svm_node *, const svm_node *);

double accuracyScore(const int, const double *, const double *);
double **confusionMatrix(const int, const double*, const double*);

void classificationReport(const int, const double *, const double *);

std::vector<int> randomChoice(const int, const int);
#endif /* Util_hpp */
