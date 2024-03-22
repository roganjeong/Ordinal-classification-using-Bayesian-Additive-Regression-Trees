// MCMC_utils.h
#pragma once

#include<Rcpp.h>


#include<iostream>
#include<cmath>


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

// void sorting_Z(NumericVector& Z, const NumericVector& Y, const int& K, const int& n)
// {
//     NumericVector temp(n);
//     // int len = sizeof(Z)/sizeof(Z[0]);
//     std::sort(Z, Z + len);//Sorting demo array
// }


void update_R(NumericVector& R, const NumericVector& Z, const NumericMatrix& Tree, const int t)
{
    NumericVector mu = rowSums(Tree);
    for (int i = 0; i < Tree.nrow(); i++)
    {
        R(i) = Z(i) - mu(i) + Tree(i, t);
    }
}


void initial_alphas(NumericVector& alphas, const int& K)
{
    Environment stats = Environment::namespace_env("stats");
    Function qnorm = stats["qnorm"];
    for (int i = 0; i < K-1; i++)
    {
        double quantile = (i + 1.0)/K;
        NumericVector z = qnorm(quantile, 0, 1);
        alphas(i) = z(0);
    }
}


// void initial_Z(NumericVector& Z, const NumericVector& Y, const int& K, const NumericVector& alphas)
// {
//     Environment EnvStats = Environment::namespace_env("EnvStats");
//     Function rnormTrunc = EnvStats["rnormTrunc"];
//     const double Inf = INFINITY;
//     const double negative_Inf = Inf*(-1);
//     const int n = Y.length();
//     NumericVector sample_drawn(1);

//     for (int i = 0; i < n; i++)
//     {
//         if (Y(i) == K)
//         {
//             sample_drawn = rnormTrunc(1, 0, 1, alphas(K-2), Inf);
//             Z(i) = sample_drawn(0); // alphas(K-2) == alpha_K-1
//         }
//         else if (Y(i) == 1)
//         {
//             sample_drawn = rnormTrunc(1, 0, 1, negative_Inf, alphas(0));
//             Z(i) = sample_drawn(0);
//         }
//         else
//         {
//             sample_drawn = rnormTrunc(1, 0, 1, alphas(Y(i)-2), alphas(Y(i)-1));
//             Z(i) = sample_drawn(0);
//         }
//     }

// }

// rnormTrunc(3,10,2,8,13) # drawing 3 samples from the truncated normal distribution with the parameters (mean=10, sd=2, min=8, max=13)

// void update_Z(NumericVector& Z, const NumericVector& Y, const int& K, const NumericVector& alphas, const NumericMatrix& Tree)
// {
    
//     Environment EnvStats = Environment::namespace_env("EnvStats");
//     Function rnormTrunc = EnvStats["rnormTrunc"];
    
//     NumericVector mu = rowSums(Tree);
//     const double Inf = INFINITY;
//     const double negative_Inf = Inf*(-1);

//     // try to use something big other than Infinity since it causes an error
//     // const double pseudo_inf = 10;

//     NumericVector sample_drawn(1);
//     for (int i = 0; i < Tree.nrow(); i++)
//     {
//         if (Y(i) == K)
//         {
//             if ( alphas(K-2) - mu(i) > 6)
//             {
//                 double adder = Rcpp::runif(1, 0, 0.3)(0);
//                 sample_drawn = alphas(K-2) + adder;
//             }
//             else
//             {
//                 sample_drawn = rnormTrunc(1, mu(i), 1, alphas(K-2), Inf); // Inf --> 
//                 Z(i) = sample_drawn(0); // alphas(K-2) == alpha_K-1
//             }
//         }
//         else if (Y(i) == 1)
//         {
//             sample_drawn = rnormTrunc(1, mu(i), 1, negative_Inf, alphas(0));
//             Z(i) = sample_drawn(0);
//         }
//         else
//         {
//             if (alphas(Y(i)-2) - mu(i) > 6)
//             {
//                 double adder = Rcpp::runif(1, 0, 0.3)(0);
//                 sample_drawn = alphas(K-2) + adder;
//             }else
//             {
//                 sample_drawn = rnormTrunc(1, mu(i), 1, alphas(Y(i)-2), alphas(Y(i)-1));
//                 Z(i) = sample_drawn(0);
//             }
//         }
//     }
// }

// Z used in the below code comes from the updated Z as the result of update_Z()

// void update_LBs(NumericVector& LBs, const NumericVector& Z, const NumericVector& Y, const int& K, const int& n, const NumericVector& alphas, const double& delta)
// {
//     const double negative_delta = delta*(-1);
//     NumericVector max_each_cat(K);

//     for (int i = 0; i < K; i++)
//     {
//         // NumericVector temp1 = which_id(Y, i+1);

//         NumericVector temp1;

//         for (int j = 0; j < n; j++)
//         {
//             if (Y(j) == i + 1.0)
//             {
//                 temp1.push_back(j);
//             }
//         }

//         int temp1_len = temp1.length();
//         NumericVector temp2(temp1_len);
//         for (int h = 0; h < temp1_len; h++)
//         {
//             int ind = temp1(h);
//             temp2(h) = Z(ind);
//         }
//         double temp3 = Rcpp::max(temp2);
//         max_each_cat(i) = temp3;
//     }

//     for (int i = 0; i < K-1; i++) // alphas = (a1, a2, a3), K = 4
//     {
//         if (i == 0)
//         {
//             NumericVector temp;
//             temp.push_back(negative_delta);
//             temp.push_back(alphas(i+1) - delta);
//             temp.push_back(max_each_cat(i));
//             LBs(i) = Rcpp::max(temp);
//         }
//         else if (i == K-2)
//         {
//             NumericVector temp(2);
//             temp.push_back(alphas(i-1));
//             temp.push_back(max_each_cat(i));
//             LBs(i) = Rcpp::max(temp);
//         }
//         else
//         {
//             NumericVector temp;
//             temp.push_back(alphas(i-1));
//             temp.push_back(alphas(i+1) - delta);
//             temp.push_back(max_each_cat(i));
//             // Rcout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
//             // Rcout << "temp : " << temp << std::endl;
//             // Rcout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
//             LBs(i) = Rcpp::max(temp);
//         }
//     }
// }

// // Z used in the below code comes from the updated Z as the result of update_Z()

// void update_UBs(NumericVector& UBs, const NumericVector& Z, const NumericVector& Y, const int& K, const int& n, const NumericVector& alphas, const double& delta)
// {
//     NumericVector min_each_cat(K); // previously : NumericVector min_each_cat(K)

//     for (int i = 0; i < K; i++) // i = 3에서 에러가 발생하고
//     {
//         // NumericVector temp1 = which_id(Y, i+1);
//         NumericVector temp1;

//         for (int j = 0; j < n; j++)
//         {
//             if (Y(j) == i + 1.0)
//             {
//                 temp1.push_back(j);
//             }
//         }

//         int temp1_len = temp1.length();
//         NumericVector temp2(temp1_len);
//         for (int h = 0; h < temp1_len; h++)
//         {
//             int ind = temp1(h);
//             temp2(h) = Z(ind);
//         }
        
//         double temp3 = Rcpp::min(temp2);
//         min_each_cat(i) = temp3;
//     } 


//     for (int i = 0; i < K-1; i++)
//     {
//         if (i == 0)
//         {
//             NumericVector temp;
//             temp.push_back(delta);
//             temp.push_back(alphas(i+1));
//             temp.push_back(min_each_cat(i+1));
//             UBs(i) = Rcpp::min(temp);
//         }
//         else
//         {
//             NumericVector temp;
//             temp.push_back(alphas(i-1) + delta);
//             if (i == K-2)
//             {
//                 const double Inf = INFINITY;
//                 temp.push_back(Inf);
//             }
//             else
//             {
//                 temp.push_back(alphas(i+1));    
//             }
//             temp.push_back(min_each_cat(i+1));
//             UBs(i) = Rcpp::min(temp);
//         }
//     }
// }
// alphas = {alphas(0), alphas(1), alphas(2)} 이렇게 되어 있는 벡터인데, 

// // Z, UB, LB used in the below code comes from the updated versions 
// // R::runif(n = 1, min = 0, max = 1)

// void update_alphas(NumericVector& alphas, const int& K, const NumericVector& UBs, const NumericVector& LBs)
// {
//     for (int i = 0; i < K-1; i++)
//     {
//         // Rcout << "alphas" << " - " << i << std::endl;
//         // Rcout << LBs(i) << "  " << UBs(i) << std::endl;

//         alphas(i) = Rcpp::runif(1, LBs(i), UBs(i))(0);
//     }
//     // Rcout << "alphas = (" << alphas << ")"<< std::endl;
// }


void compute_posterior_mean(const NumericMatrix& post_Z, NumericVector& post_mean_predicted, const int& n_test)
{
    
    for (int i = 0; i < n_test; i++)
    {
        post_mean_predicted(i) = mean(post_Z(i,_));
        // post_mean_LBs(i, post_iter) = mean(post_LBs(i,_));
        // post_mean_UBs(i, post_iter) = mean(post_UBs(i,_));
        // post_mean_alphas(i, post_iter) = mean(post_alphas(i,_));
    }
}


// void check_nan_Z(const NumericVector& Z, const int n)
// {
//     for (int i = 0; i < n; i++)
//     {
//         if (isnan(Z(i)) == true)
//         {
//             break;
//         }
//     }
// }

// void check_nan_LBs(const NumericVector& LBs, const int K)
// {
//     for (int i = 0; i < K-1; i++)
//     {
//         if (isnan(LBs(i)) == true)
//         {
//             break;
//         }
//     }
// }

// void check_nan_UBs(const NumericVector UBs, const int K)
// {
//     for (int i = 0; i < K-1; i++)
//     {
//         if (isnan(UBs(i)) == true)
//         {
//             break;
//         }
//     }
// }

// void check_nan_alphas(const NumericVector alphas, const int K)
// {
//     for (int i = 0; i < K-1; i++)
//     {
//         if (isnan(alphas(i)) == true)
//         {
//             break;
//         }
//     }
// }

// need to fix
// void find_mode(NumericVector predicted_Y, const NumericMatrix& post_predicted, const int& K, const int& n_test, const int& n_post)
// {
//     IntegerMatrix count(n_test,K);
//     for (int p = 0; p < K; p++)
//     {
//         count(_, p) = rep(0, n_test);
//     }

//     for (int t = 0; t < n_test; t++)
//     {
//         IntegerVector counter = rep(0, K);
//         for (int i = 0; i < n_post; i++)
//         {
//             for (int j = 1; j <= K; j++)
//             {
//                 if (post_predicted(t, i) == j)
//                 {
//                     int temp = counter(j-1);
//                     counter(j-1) = temp + 1;
//                 }
//             }
//         }

//         int max_cat_count = Rcpp::max(counter);
//         for (int i = 0; i < K; i++)
//         {
//             if (counter(i) == max_cat_count)
//             {
//                 predicted_Y(t) = i + 1.0;
//             }
//         }
//     }

// }