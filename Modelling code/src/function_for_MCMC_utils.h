// function_for_MCMC_utils.h
#pragma once

#include <Rcpp.h>


using namespace Rcpp;


NumericVector which_id(const NumericVector& vector, int& value)
{
    int n = vector.length();
    NumericVector temp;

    for (int i = 0; i < n; i++)
    {
        if (vector(i) == value + 0.0)
        {
            temp.push_back(i);
        }
    }
    return temp;
}