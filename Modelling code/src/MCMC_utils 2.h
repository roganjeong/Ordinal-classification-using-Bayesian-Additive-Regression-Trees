// MCMC_utils.h
#pragma once

#include<Rcpp.h>

#include<iostream>
#include<cmath>

#include "functions.h" // error might occurs because of this line


using namespace Rcpp;
using namespace std; // to use the value of infinity


IntegerVector init_seq(const int n_iter, const int thin, const int burn_in)
{
    IntegerVector seq;
    for (int i = 0; i < n_iter; i++)
    {
        int temp = i - n_iter / 2;
        if (i >= burn_in && temp % thin == 0)
        {
            seq.push_back(i);
        }
    }
    return seq;
}

void log_with_LB(NumericVector& res, const NumericVector& src)
{
    const double LB     = pow(0.1, 300);
    const double log_LB = -300 * log(10);
    for (int i = 0; i < src.length(); i++)
    {
        if (src(i) > LB)
            res(i) = log(src(i));
        else
            res(i) = log_LB;
    }
}

NumericVector rowSums_without(const NumericMatrix& src, const int idx)
{
    NumericVector res = rowSums(src);
    for (int i = 0; i < src.nrow(); i++)
    {
        res(i) -= src(i, idx);
    }
    return res;
}

void update_R(NumericVector& R, const NumericVector& Z, const NumericMatrix& Tree, const int t)
{
    NumericVector mu = rowSums(Tree);
    for (int i = 0; i < Tree.nrow(); i++)
    {
        R(i) = Z(i) - mu(i) + Tree(i, t);
    }
}

// rnormTrunc(3,10,2,8,13) # drawing 3 samples from the truncated normal distribution with the parameters (mean=10, sd=2, min=8, max=13)
// [[Rcpp::export]]
void update_Z(NumericVector& Z, const NumericVector& Y, const int& K, const NumericVector& alphas, const NumericMatrix& Tree)
{
    
    Environment EnvStats = Environment::namespace_env("EnvStats");
    Function rnormTrunc = EnvStats["rnormTrunc"];
    
    NumericVector mu = rowSums(Tree);
    const double Inf = INFINITY;
    const double negative_Inf = Inf*(-1);
    NumericVector sample_drawn(1);
    for (int i = 0; i < Tree.nrow(); i++)
    {
        if (Y(i) == K)
        {
            sample_drawn = rnormTrunc(1, mu(i), 1, alphas(K-2), Inf);
            Z(i) = sample_drawn(0); // alphas(K-2) == alpha_K-1
        }
        else if (Y(i) == 1)
        {
            sample_drawn = rnormTrunc(1, mu(i), 1, negative_Inf, alphas(0));
            Z(i) = sample_drawn(0);
        }
        else
        {
            sample_drawn = rnormTrunc(1, mu(i), 1, alphas(Y(i)-2), alphas(Y(i)-1));
            Z(i) = sample_drawn(0);
        }
    }
}

// Z used in the below code comes from the updated Z as the result of update_Z()
// [[Rcpp::export]]
void update_LBs(NumericVector& LBs, const NumericVector& Z, const NumericVector& Y, const int& K, const NumericVector& alphas, const double& delta)
{
    const double negative_delta = delta*(-1);

    NumericVector max_each_cat(K);

    for (int i = 0; i < K; i++)
    {
        NumericVector temp1 = which(Y, i+1);
        int temp1_len = temp1.length();
        NumericVector temp2(temp1_len);
        for (int h = 0; h < temp1_len; h++)
        {
            double ind = temp1(h);
            temp2(h) = Y(ind);
        }
        double temp3 = Rcpp::max(temp2);
        max_each_cat(i) = temp3;
    }

    for (int i = 0; i < K-1; i++)
    {
        if (i == 0)
        {
            NumericVector temp(3);
            temp.push_back(negative_delta);
            temp.push_back(alphas(i+1) - delta);
            temp.push_back(max_each_cat(i));
            LBs(i) = Rcpp::max(temp);
        }
        else if (i == K-2)
        {
            NumericVector temp(2);
            temp.push_back(alphas(i));
            temp.push_back(max_each_cat(i+1));
            LBs(i) = Rcpp::max(temp);
        }
        else
        {
            NumericVector temp(3);
            temp.push_back(alphas(i-1));
            temp.push_back(alphas(i+1) - delta);
            temp.push_back(max_each_cat(i));
            LBs(i) = Rcpp::max(temp);
        }
    }
}

// Z used in the below code comes from the updated Z as the result of update_Z()
// [[Rcpp::export]]
void update_UBs(NumericVector& UBs, const NumericVector& Z, const NumericVector& Y, const int& K, const NumericVector& alphas, const double& delta)
{
    NumericVector min_each_cat(K);

    for (int i = 0; i < K; i++)
    {
        NumericVector temp1 = which(Y, i+1);
        int temp1_len = temp1.length();
        NumericVector temp2(temp1_len);
        for (int h = 0; h < temp1_len; h++)
        {
            double ind = temp1(h);
            temp2(h) = Y(ind);
        }
        double temp3 = Rcpp::min(temp2);
        min_each_cat(i) = temp3;
    }

    for (int i = 0; i < K-1; i++)
    {
        if (i == 0)
        {
            NumericVector temp(3);
            temp.push_back(delta);
            temp.push_back(alphas(i+1));
            temp.push_back(min_each_cat(i+1));
            UBs(i) = Rcpp::min(temp);
        }
        else
        {
            NumericVector temp(3);
            temp.push_back(alphas(i-1) + delta);
            temp.push_back(alphas(i+1));
            temp.push_back(min_each_cat(i+1));
            UBs(i) = Rcpp::min(temp);
        }
    }
}

// Z, UB, LB used in the below code comes from the updated versions 
// R::runif(n = 1, min = 0, max = 1)

void update_alphas(NumericVector& alphas, const int& K, const NumericVector& UBs, const NumericVector& LBs)
{
    for (int i = 0; i < K-1; i++)
    {
        alphas(i) = Rcpp::runif(1, LBs(i), UBs(i))(0);
    }
}