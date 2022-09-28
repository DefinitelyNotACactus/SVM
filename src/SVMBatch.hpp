//
//  SVMBatch.hpp
//  SVM
//
//  Created by David Galvao on 19/09/22.
//

#ifndef SVMBatch_hpp
#define SVMBatch_hpp

#include "AbstractSVM.hpp"
#include "SVMParameters.hpp"

#include <tuple>
#include <unordered_set>

class SVMBatch : public AbstractSVM {
public:
    SVMBatch(svm_parameter *, double, double);
    
    void fit(const svm_problem &);
    
private:
    double pct, epsilon;
    
    std::tuple<svm_problem, svm_problem> separate_classes(const svm_problem &);
    void removeSamples(const std::unordered_set<int> &, svm_problem &);
    void getCandidates(std::unordered_set<int> &, std::unordered_set<int> &, std::unordered_set<int> &, const double *, SVMParameters &, const svm_problem &);
    void selectCandidates(const std::unordered_set<int> &, std::unordered_set<int> &, const svm_problem &, svm_problem &);
};
#endif /* SVMBatch_hpp */
