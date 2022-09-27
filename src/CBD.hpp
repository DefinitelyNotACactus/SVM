//
//  CBD.hpp
//  SVM
//
//  Created by David Galvao on 24/09/22.
//

#ifndef CBD_hpp
#define CBD_hpp

#include "AbstractSVM.hpp"

#include <tuple>

struct CBDScore {
    int index;
    double score;
    
    CBDScore() { };
    CBDScore(int index, double score) : index(index), score(score) { };
    
    bool operator > (const CBDScore &a) const { return score > a.score; }
    bool operator < (const CBDScore &a) const { return score < a.score; }
    
};

class CBD : public AbstractSVM {
public:
    CBD(svm_parameter *, int, double);
    
    void fit(const svm_problem &);
    
private:
    const int K;
    const double G;
    
    CBDScore *determineScores(int, const svm_problem &);
    double computeDistance(const std::vector<double> &, const std::vector<double> &);
    std::tuple<double **, int **> computeNeighbors(const int, const svm_problem &);
};
#endif /* CBD_hpp */
