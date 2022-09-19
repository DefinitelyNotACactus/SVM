//
//  Util.hpp
//  SVM
//
//  Created by David Galvao on 18/09/22.
//

#ifndef Util_hpp
#define Util_hpp

#include "svm.hpp"

double dot(const int, const svm_node *, const svm_node *);
double dot(const int, const double *, const double *);
double norm(const int, const double *);
double *diff(const int, const svm_node *, const svm_node *);

#endif /* Util_hpp */
