#include<Rcpp.h>
#include<iostream>
#include<cmath>

using namespace Rcpp;
using namespace std;


// List test_printing(
//     const double test = 1.0
// ) {
//     const double Infy = INFINITY;
//     const double negative_Infy = round_toward_neg_infinity*(-1);
//     double r = Rcpp::rnorm(1, 0, 1)(0);
//     NumericVector vec_one = {1, r, Infy, negative_Infy};

//     List L = List::create(
//         Named("test") = vec_one(0),
//         Named("r") = vec_one(1),
//         Named("Inf") = vec_one(2),
//         Named("Nagative_Inf") = vec_one(3)
//     );
//     
//     return L;
// }

// [[Rcpp::export]]
List test_printing(
    double test = 1.0
) {
    const double Infy = INFINITY;
    NumericVector vec_one = {test, Infy, 3};

    if (test == 1)
    {
        vec_one(0) = test + 1;
    }
    else
    {
        vec_one(0) = test - 1;
    }

    List L = List::create(
        Named("test") = vec_one(0) + 1, // 3
        Named("Inf") = vec_one(1) * (-1),
        Named("last") = vec_one(2)
    );

    return L;
}