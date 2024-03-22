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


void initial_alphas(NumericVector& alphas, const NumericVector& Y, const int& K)
{
    Environment stats = Environment::namespace_env("stats");
    Function qnorm = stats["qnorm"];

    const int n = Y.length();

    NumericVector quantile = rep(0.0,K-1);
    
    for (int i = 1; i < K; i++)
    {
        double count = 0.0;
        for (int j = 0; j < n; j++)
        {
            if (Y(j) == i)
            {
                double temp = count + 1.0;
                count = temp;
            }
        }
        if (i == 1)
        {
            quantile(i-1) = (count / n);
        }
        else
        {
            quantile(i-1) = (count / n) + quantile(i-2);
        }
        // Rcout << quantile << std::endl;
        NumericVector z = qnorm(quantile(i-1), 0, 1);
        alphas(i-1) = z(0);
    }
}


void initial_Z(NumericVector& Z, const NumericVector& Y, const int& K, const NumericVector& alphas)
{
    
    Environment truncnorm = Environment::namespace_env("truncnorm");
    Function rtruncnorm = truncnorm["rtruncnorm"];
    const double Inf = INFINITY;
    const double negative_Inf = Inf*(-1);
    const int n = Y.length();
    NumericVector sample_drawn(1);


    for (int i = 0; i < n; i++)
    {
        if (Y(i) == K)
        {
            sample_drawn = rtruncnorm(1, alphas(K-2), Inf, 0, 1);
            Z(i) = sample_drawn(0); // alphas(K-2) == alpha_K-1
        }
        else if (Y(i) == 1)
        {
            sample_drawn = rtruncnorm(1, negative_Inf, alphas(0), 0, 1);
            Z(i) = sample_drawn(0);
        }
        else
        {
            sample_drawn = rtruncnorm(1, alphas(Y(i)-2), alphas(Y(i)-1), 0, 1);
            Z(i) = sample_drawn(0);
        }
    }


    // for (int i = 0; i < n; i++)
    // {
    //     if (Y(i) == K)
    //     {
    //         sample_drawn = rtruncnorm(1, alphas(K-2), Inf, 0, 1);
    //         Z(i) = sample_drawn(0); // alphas(K-2) == alpha_K-1
    //     }
    //     else if (Y(i) == 1)
    //     {
    //         sample_drawn = rtruncnorm(1, negative_Inf, alphas(0), 0, 1);
    //         Z(i) = sample_drawn(0);
    //     }
    //     else
    //     {
    //         sample_drawn = rtruncnorm(1, alphas(Y(i)-2), alphas(Y(i)-1), 0, 1);
    //         Z(i) = sample_drawn(0);
    //     }
    // }

}

// rnormTrunc(3,10,2,8,13) # drawing 3 samples from the truncated normal distribution with the parameters (mean=10, sd=2, min=8, max=13)
// [[Rcpp::export]]
void update_Z(NumericVector& Z, const NumericVector& Y, const int& K, const NumericVector& alphas, const NumericMatrix& Tree)
{
    
    Environment truncnorm = Environment::namespace_env("truncnorm");
    Function rtruncnorm = truncnorm["rtruncnorm"];
    
    NumericVector mu = rowSums(Tree);
    const double Inf = INFINITY;
    const double negative_Inf = Inf*(-1);

    // try to use something big other than Infinity since it causes an error
    // const double pseudo_inf = 10;
    const int n = Y.length();
    NumericVector sample_drawn(1);
    for (int i = 0; i < n; i++)
    {
        if (Y(i) == K)
        {
            sample_drawn = rtruncnorm(1, alphas(K-2), Inf, mu(i), 1);
            Z(i) = sample_drawn(0); // alphas(K-2) == alpha_K-1
            
        }
        else if (Y(i) == 1)
        {
            sample_drawn = rtruncnorm(1, negative_Inf, alphas(0), mu(i), 1);
            Z(i) = sample_drawn(0);
        }
        else
        {
            sample_drawn = rtruncnorm(1, alphas(Y(i)-2), alphas(Y(i)-1), mu(i), 1);
            Z(i) = sample_drawn(0);
        }
    }
}

// Z used in the below code comes from the updated Z as the result of update_Z()
// [[Rcpp::export]]
void update_LBs(NumericVector& LBs, const NumericVector& Z, const NumericVector& Y, const int& K, const int& n, const NumericVector& alphas) // const double& delta
{
    // const int negative_delta = delta*(-1);
    NumericVector max_each_cat(K);

    for (int i = 0; i < K; i++)
    {
        // NumericVector temp1 = which_id(Y, i+1);

        IntegerVector temp1;

        for (int j = 0; j < n; j++)
        {
            if (Y(j) == i + 1.0)
            {
                temp1.push_back(j);
            }
        }

        int temp1_len = temp1.length();
        NumericVector temp2(temp1_len);
        for (int h = 0; h < temp1_len; h++)
        {
            int ind = temp1(h);
            temp2(h) = Z(ind);
        }
        double temp3 = Rcpp::max(temp2);
        max_each_cat(i) = temp3;
        
    }
    

    for (int i = 0; i < K-1; i++) // alphas = (a1, a2, a3), K = 4
    {
        LBs(i) = max_each_cat(i);
        // if (i == 0)
        // {
        //     NumericVector temp(3);
        //     temp(0) = negative_delta;
        //     temp(1) = alphas(i+1) - delta;
        //     temp(2) = max_each_cat(i);
        //     // temp.push_back(negative_delta);
        //     // temp.push_back(alphas(i+1) - delta);
        //     // temp.push_back(max_each_cat(i));
        //     LBs(i) = Rcpp::max(temp);
        // }
        // else if (i == K-2)
        // {
        //     NumericVector temp(2);
        //     temp(0) = alphas(i-1);
        //     temp(1) = max_each_cat(i);
        //     // temp.push_back(alphas(i-1));
        //     // temp.push_back(max_each_cat(i));
        //     LBs(i) = Rcpp::max(temp);
        // }
        // else
        // {
        //     NumericVector temp(3);
        //     temp(0) = alphas(i-1);
        //     temp(1) = alphas(i+1) - delta;
        //     temp(2) = max_each_cat(i);
        //     // temp.push_back(alphas(i-1));
        //     // temp.push_back(alphas(i+1) - delta);
        //     // temp.push_back(max_each_cat(i)); 
        //     // Rcout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
        //     // Rcout << "temp : " << temp << std::endl;
        //     // Rcout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
        //     LBs(i) = Rcpp::max(temp);
        // }
    }
}

// Z used in the below code comes from the updated Z as the result of update_Z()
// [[Rcpp::export]]
void update_UBs(NumericVector& UBs, const NumericVector& Z, const NumericVector& Y, const int& K, const int& n, const NumericVector& alphas) // const int& delta
{
    NumericVector min_each_cat(K);
    // const double Inf = INFINITY;

    for (int i = 0; i < K; i++) // i = 3에서 에러가 발생하고
    {
        // NumericVector temp1 = which_id(Y, i+1);
        IntegerVector temp1;

        for (int j = 0; j < n; j++)
        {
            if (Y(j) == i + 1.0)
            {
                temp1.push_back(j);
            }
        }

        int temp1_len = temp1.length();
        NumericVector temp2(temp1_len);
        for (int h = 0; h < temp1_len; h++)
        {
            int ind = temp1(h);
            temp2(h) = Z(ind);
        }
        
        double temp3 = Rcpp::min(temp2);
        min_each_cat(i) = temp3;

    } 

    for (int i = 0; i < K-1; i++)
    {
        UBs(i) = min_each_cat(i+1);

        // if (i == 0)
        // {
        //     NumericVector temp(3);
        //     temp(0) = delta;
        //     temp(1) = alphas(i+1);
        //     temp(2) = min_each_cat(i+1);
        //     // temp.push_back(delta);
        //     // temp.push_back(alphas(i+1));
        //     // temp.push_back(min_each_cat(i+1));
        //     UBs(i) = Rcpp::min(temp);
        // }
        // else
        // {
        //     NumericVector temp(2);
        //     temp.push_back(alphas(i-1) + delta);
        //     if (i == K-2)
        //     {
        //         temp(0) = Inf;
        //         // temp.push_back(Inf);
        //     }
        //     else
        //     {
        //         temp(0) = alphas(i+1);
        //         // temp.push_back(alphas(i+1));    
        //     }
        //     temp(1) = min_each_cat(i+1);
        //     // temp.push_back(min_each_cat(i+1));
        //     UBs(i) = Rcpp::min(temp);
        // }
    }
}


void update_alphas(NumericVector& alphas, const NumericMatrix& Tree, const int& K, const NumericVector& Y, int& accepted)
{
    Environment truncnorm = Environment::namespace_env("truncnorm");
    Function rtruncnorm = truncnorm["rtruncnorm"];

    Environment stats = Environment::namespace_env("stats");
    Function integrate = stats["integrate"];
    Function dnorm = stats["dnorm"];

    const double Inf = INFINITY;
    const double negative_Inf = Inf*(-1);
    const int n = Y.length();
    NumericVector sample_drawn(1);
    NumericVector alphas_prop = rep(0.0, K-1);

    NumericVector Integrations_past = rep(0.0, n);
    NumericVector Integrations_prop = rep(0.0, n);

    double sigma_gamma = 0.71;

    // defining proposal alphas
    for (int i = 0; i < K-1; i++)
    {
        if (i == 0)
        {
            sample_drawn = rtruncnorm(1, negative_Inf, alphas(1), alphas(0), sigma_gamma);
            alphas_prop(i) = sample_drawn(0);
        }
        else if (i == K-2)
        {
            sample_drawn = rtruncnorm(1, alphas_prop(K-3), Inf, alphas(K-2), sigma_gamma);
            alphas_prop(i) = sample_drawn(0);
        }
        else
        {
            sample_drawn = rtruncnorm(1, alphas_prop(i-1), alphas(i+1), alphas(i), sigma_gamma);
            alphas_prop(i) = sample_drawn(0);
        }
    }

    
    NumericVector mu = rowSums(Tree);
    double acceptance_ratio = 1.0;
    double multiplier = 1.0;
    List temp;
    // calculating acceptance ratio
    for (int i = 0; i < n; i++)
    {
        // Rcout << "i : [" << i << "]"<< std::endl;
        // Function dnorm = stats["dnorm"];
        if ( Y(i) == 1)
        {
            temp = integrate(dnorm, negative_Inf, alphas(0), mu(i), 1);
            Integrations_past(i) = temp["value"];

            temp = integrate(dnorm, negative_Inf, alphas_prop(0), mu(i), 1);
            Integrations_prop(i) = temp["value"];
        }
        else if( Y(i) == K )
        {
            temp = integrate(dnorm, alphas(K-2), Inf, mu(i), 1);
            Integrations_past(i) = temp["value"];

            temp = integrate(dnorm, alphas_prop(K-2), Inf, mu(i), 1);
            Integrations_prop(i) = temp["value"];
        }
        else
        {
            // Rcout << "1" << std::endl;
            temp = integrate(dnorm, alphas(Y(i)-2), alphas(Y(i)-1), mu(i), 1);
            // Rcout << "2" << std::endl;
            Integrations_past(i) = temp["value"];

            // Rcout << "3" << std::endl;
            temp = integrate(dnorm, alphas_prop(Y(i)-2), alphas_prop(Y(i)-1), mu(i), 1);
            // Rcout << "4" << std::endl;
            Integrations_prop(i) = temp["value"];
        }
        multiplier = Integrations_prop(i) / Integrations_past(i);
    }

    double norm_past = 0.0;
    double norm_prop = 0.0;
    double adder;
    for (int i = 0; i < K-1; i++)
    {
        adder = norm_past;
        norm_past = adder + alphas(i)*alphas(i);

        adder = norm_prop;
        norm_prop = adder + alphas_prop(i)*alphas_prop(i);
    }
    acceptance_ratio = multiplier * exp((norm_past - norm_prop)/(2*sigma_gamma*sigma_gamma));

    // reject or accept
    double u = Rcpp::runif(1,0,1)(0);
    int adding_1;
    if ( acceptance_ratio >= 1 || acceptance_ratio >= u )
    {
        alphas = alphas_prop;
        adding_1 = accepted;
        accepted = 1 + adding_1;
    }



}


void compute_posterior_mean(const int& K, const int& post_iter, const NumericMatrix post_LBs, NumericVector post_mean_LBs, const NumericMatrix post_UBs, NumericVector post_mean_UBs, const NumericMatrix post_alphas, NumericVector post_mean_alphas)
{
    
    for (int i = 0; i < K-1; i++)
    {
        post_mean_LBs(i, post_iter) = mean(post_LBs(i,_));
        post_mean_UBs(i, post_iter) = mean(post_UBs(i,_));
        post_mean_alphas(i, post_iter) = mean(post_alphas(i,_));
    }
}


void check_nan_Z(const NumericVector& Z, const int n)
{
    for (int i = 0; i < n; i++)
    {
        if (isnan(Z(i)) == true)
        {
            break;
        }
    }
}

void check_nan_LBs(const NumericVector& LBs, const int K)
{
    for (int i = 0; i < K-1; i++)
    {
        if (isnan(LBs(i)) == true)
        {
            break;
        }
    }
}

void check_nan_UBs(const NumericVector UBs, const int K)
{
    for (int i = 0; i < K-1; i++)
    {
        if (isnan(UBs(i)) == true)
        {
            break;
        }
    }
}

void check_nan_alphas(const NumericVector alphas, const int K)
{
    for (int i = 0; i < K-1; i++)
    {
        if (isnan(alphas(i)) == true)
        {
            break;
        }
    }
}


void find_mode(NumericVector predicted_Y, const NumericMatrix& post_predicted, const int& K, const int& n_test, const int& n_post)
{
    IntegerMatrix count(n_test,K);
    for (int p = 0; p < K; p++)
    {
        count(_, p) = rep(0, n_test);
    }

    for (int t = 0; t < n_test; t++)
    {
        IntegerVector counter = rep(0, K);
        for (int i = 0; i < n_post; i++)
        {
            for (int j = 1; j <= K; j++)
            {
                if (post_predicted(t, i) == j)
                {
                    int temp = counter(j-1);
                    counter(j-1) = temp + 1;
                }
            }
        }

        int max_cat_count = Rcpp::max(counter);
        for (int i = 0; i < K; i++)
        {
            if (counter(i) == max_cat_count)
            {
                predicted_Y(t) = i + 1.0;
            }
        }
    }
}

// need to be fixed; only prints a matrix with elements = 0
void eachcat_probability(NumericMatrix probability_categories, const NumericMatrix& post_predicted, const int& K, const int& n_test, const int& n_post)
{
    
    for (int i = 0; i < n_test; i++)
    {
        NumericVector count = rep(0.0, K);
        NumericVector prob = rep(0.0, K);
        for (int j = 0; j < n_post; j++)
        {
            for (int k = 0; k < K; k++)
            {
                if (post_predicted(i,j) == k+1)
                {
                    double added = count(k) + 1.0;
                    count(k) = added;
                }
            }
        }

        for (int k = 0; k < K; k++)
        {
            double counted = count(k);
            prob(k) = counted/n_post;
            probability_categories(i,k) = prob(k);
        }
        // probability_categories(_,i) = prob;
    }

}