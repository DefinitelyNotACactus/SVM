//
//  SVMParameters.hpp
//  SVM
//
//  Created by David Galvao on 18/09/22.
//

#ifndef SVMParameters_hpp
#define SVMParameters_hpp

#include "svm.hpp"

#include <unordered_set>

class SVMParameters {
public:
    SVMParameters(const svm_model *, const svm_problem &, bool margin=false);
    ~SVMParameters();
    
    bool isSV(const int);
    double computeLeftHand(const svm_node *, double);
    void print();
    
private:
    const svm_model *model;
    int l, N, d, lenSV0, lenSV1;
    std::unordered_set<int> indicesSV;
    svm_node **X, **XSV0, **XSV1, *XSV;
    double *y, *alpha, sumB, ysv, margin;

    double kernel(const svm_node *, const svm_node *);
    
    void computeB();
    void computeMargin();
    void separateVS();
};
#endif /* SVMParameters_hpp */
