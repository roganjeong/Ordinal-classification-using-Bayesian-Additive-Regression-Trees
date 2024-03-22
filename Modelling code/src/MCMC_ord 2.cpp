#include <Rcpp.h>

#include<iostream>
#include<cmath>

#include "decision_tree.h"
#include "MCMC_utils.h"

using namespace Rcpp;
using namespace std;


// [[Rcpp::export]]
List MCMC_ord(
    const NumericMatrix& Xpred,
    const NumericVector& Y,  // original : IntegerVector - will be changed to NumericVector in purpose of debugging
    const int K,
    const NumericMatrix& Xpred_test,
    const NumericVector& Y_test, // The datatype should be the same as that of 'Y'
    const double p_grow,   // Prob. of GROW
    const double p_prune,  // Prob. of PRUNE
    const double p_change, // Prob. of CHANGE
    const int m,           // Num. of Trees: default setting 100
    double dir_alpha, double alpha, double beta,
    const int n_iter,
    const bool verbose = false
) {
    
    // Data preparation
    const int P = Xpred.ncol(); // number of covariates
    const int n = Xpred.nrow(); // number of observations
    const int n_test = Xpred_test.nrow();

    NumericVector Xcut[P]; // e.g. unique value of potential confounders
    for (int j = 0; j < P; j++)
    {
        NumericVector temp = unique(Xpred(_, j));
        temp.sort();
        Xcut[j] = temp;
    }

    // Initial Setup
    // Priors, initial values and hyper-parameters
    NumericVector Z = Rcpp::rnorm(n, 0, 1); // latent variable - original :  Z = Rcpp::rnorm(n, R::qnorm(mean(Y_trt), 0, 1, true, false), 1)
    NumericVector prob = {p_grow, p_prune, p_change};

    double sigma2 = 1.0;
    //NumericVector dir_alpha_hist (n_iter + 1); // create placeholder for dir_alpha
    //dir_alpha_hist(0) = dir_alpha;

    // sigma_mu based on min/max of Y, Y (A=1) and Y (A=0)
    double sigma_mu   = (Rcpp::max(Z) - Rcpp::min(Z))/(4*sqrt(m)); // original : std::max(pow(min(Z)     / (-2 * sqrt(m)), 2), pow(max(Z)     / (2 * sqrt(m)), 2))
    // 4 = 2*2 = 2*k, k is fixed as 2

    // Initial values of R
    NumericVector R  = clone(Z);

    // Initial values for the selection probabilities
    NumericVector post_dir_alpha  = rep(1.0, P);

    // thin = 10, burn-ins = n_iter/2
    //int thin       = 10;
    //int burn_in    = n_iter / 2;
    // int n_post     = (n_iter - burn_in) / thin; // number of post sample

    NumericVector predicted_Y(n_test); // must be changed to 'IntegerVector' - now it should be changed to NumericVector
    // IntegerMatrix ind    (n_post, P);    // original : (n_post, P+1)

    IntegerMatrix Obs_list(n, m); // changed list to matrix

    // Place-holder for the posterior samples
    NumericMatrix Tree   (n, m);

    DecisionTree dt_list[m]; // changed to array of trees
    for (int t = 0; t < m; t++)
    {
        dt_list[t]  = DecisionTree(n, t);
    }

    // Obtaining namespace of MCMCpack package
    Environment MCMCpack = Environment::namespace_env("MCMCpack");

    // Picking up rdirichlet() function from MCMCpack package
    Function rdirichlet = MCMCpack["rdirichlet"];

    NumericVector prop_prob = rdirichlet(1, rep(dir_alpha, P));   // original : rep(dir_alpha, P + 1)

    // added arguments : delta, alphas, LBs, UBs, 
    NumericVector mu = rowSums(Tree);
    const double delta = abs(Rcpp::mean(mu)) + 3.0; // should delta never be updated in MCMC...?
    const double negative_delta = delta*(-1);
    NumericVector LBs = rep(0.0, K-1);
    NumericVector UBs = rep(0.0, K-1);
    NumericVector alphas = rep(0.0, K-1);

    for (int u = 0; u < K-1; u++)
    {
        if (u == 0)
        {
            alphas(u) = Rcpp::runif(1, negative_delta, delta)(0);
        }
        else
        {
            double alpha_past1 = alphas(u-1);
            alphas(u) = Rcpp::runif(1, alpha_past1, alpha_past1 + delta)(0);
        }
    }

    ////////////////////////////////////////
    //////////   Run main MCMC    //////////
    ////////////////////////////////////////

    // Run MCMC
    for (int iter = 1; iter <= n_iter; iter++)
    {
        if (verbose)
        {
            if (iter % 100 == 0)
                Rcout << "Rcpp iter : " << iter << " of " << n_iter << std::endl;
        }
        update_Z(Z, Y, K, alphas, Tree); // original : update_Z(Z, Y_trt, Tree)
        update_LBs(LBs, Z, Y, K, alphas, delta);
        update_UBs(UBs, Z, Y, K, alphas, delta);
        update_alphas(alphas, K, UBs, LBs);

        for (int t = 0; t < m; t++)
        {
            // decision trees
            update_R(R, Z, Tree, t);

            if (dt_list[t].length() == 1)
            { 
                // tree has no node yet
                // grow first step
                dt_list[t].GROW_first(
                    Xpred, Xcut, sigma2, sigma_mu, R, Obs_list,
                    p_prune, p_grow, alpha, beta, prop_prob
                );
            }
            else
            {
                int step = sample(3, 1, false, prob)(0);
                switch (step)
                {
                    case 1: // GROW step
                        dt_list[t].GROW(
                            Xpred, Xcut, sigma2, sigma_mu, R, Obs_list,
                            p_prune, p_grow, alpha, beta, prop_prob
                        );
                        break;

                    case 2: // PRUNE step
                        dt_list[t].PRUNE(
                            Xpred, Xcut, sigma2, sigma_mu, R, Obs_list,
                            p_prune, p_grow, alpha, beta, prop_prob
                        );
                        break;

                    case 3: // CHANGE step
                        dt_list[t].CHANGE(
                            Xpred, Xcut, sigma2, sigma_mu, R, Obs_list,
                            p_prune, p_grow, alpha, beta, prop_prob
                        );
                        break;

                    default: {}
                } // end of switch
            }     // end of tree instance
            dt_list[t].Mean_Parameter(Tree, sigma2, sigma_mu, R, Obs_list);
        }
        // NumericMatrix& Tree,
        // double sigma2,
        // double sigma_mu,
        // const NumericVector& R,
        // const IntegerMatrix& Obs_list

        // the codes regarding to the outcome model have been removed

        // the codes regarding to the sample variance parameter have been removed

        // Num. of inclusion of each potential confounder
        NumericVector add(P);
        for (int t = 0; t < m; t++)
        {
            add  += dt_list[t].num_included(P);
        }

        post_dir_alpha = rep(1.0, P) + add;
        prop_prob = rdirichlet(1, post_dir_alpha);

        // making predictions
        for (int i = 0; i < m; i++)
        {
            dt_list[i].Predict_ord(Tree, Xcut, Xpred_test, n_test);
        }

        NumericVector predicted_mu = rowSums(Tree);
        for (int i = 0; i < n_test; i++)
        {
            if (predicted_mu(i) >= alphas(K-2)) // alphas(K-2) == alpha_K-1
            {
                predicted_Y(i) = K;
            }
            else if (predicted_mu(i) < alphas(0))
            {
                predicted_Y(i) = 1;
            }
            else
            {
                for (int pred_ind = 0; pred_ind < K-2; pred_ind++)
                {
                    if ((alphas(pred_ind) <= predicted_mu(i)) && (predicted_mu(i) < alphas(pred_ind + 1)))
                    {
                        predicted_Y(i) = pred_ind + 1;
                    }
                }
            }
        }


        Rcpp::checkUserInterrupt(); // check for break in R

    } // end of MCMC iterations

    List L = List::create(
        Named("predicted_Y") = predicted_Y,
        Named("Y_test")      = Y_test,
        Named("cutoffs")     = alphas,
        Named("UpperBounds") = UBs,
        Named("LowerBounds") = LBs,
        Named("dir_alpha")   = post_dir_alpha
    );

    return L;
}
