#include <Rcpp.h>

#include<iostream>
#include<cmath>

#include "decision_tree.h"
#include "MCMC_utils_problemsolving.h"

using namespace Rcpp;
using namespace std;


// [[Rcpp::export]]
List MCMC_ord(
    const NumericMatrix& Xpred,
    const NumericVector& Z,
    const NumericMatrix& Xpred_test,
    const NumericVector& Z_test,
    const double p_grow,   
    const double p_prune,  
    const double p_change, 
    const int m,           
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
    

    NumericVector prob = {p_grow, p_prune, p_change};

    double sigma2 = 1.0;

    // sigma_mu based on min/max of Y, Y (A=1) and Y (A=0)
    double sigma_mu   = (Rcpp::max(Z) - Rcpp::min(Z))/(4*sqrt(m)); // original : std::max(pow(min(Z)     / (-2 * sqrt(m)), 2), pow(max(Z)     / (2 * sqrt(m)), 2))
    // 4 = 2*2 = 2*k, k is fixed as 2

    // Initial values of R
    NumericVector R  = rep(0.0, n); // clone(Z)

    // Initial values for the selection probabilities
    NumericVector post_dir_alpha  = rep(1.0, P);

    // thin = 10, burn-ins = n_iter/2
    int thin       = 10;
    int burn_in    = n_iter / 2;
    int n_post     = (n_iter - burn_in) / thin; // number of post sample
    int post_iter  = 0;
    

    NumericMatrix sample_Z(n_test, n_iter);
  
    NumericMatrix sample_dir_alpha(P, n_iter);
    

    NumericMatrix post_Z(n_test, n_post);
   
    NumericVector post_mean_predicted(n_test); // instead of "NumericVector post_mode_predicted(n_test)"

    

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

    

    ////////////////////////////////////////
    //////////   Run main MCMC    //////////
    ////////////////////////////////////////

    // Run MCMC
    for (int iter = 1; iter <= n_iter; iter++)
    {
        if (verbose)
        {
            if (iter % 100 == 0)
            {
                Rcout << "_______________________________________________________________________________" << std::endl;
                Rcout << "Rcpp iter : " << iter << " of " << n_iter << std::endl;
                Rcout << "_______________________________________________________________________________" << std::endl;
            }
        }


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
       

        // Num. of inclusion of each potential confounder
        NumericVector add(P);
        for (int t = 0; t < m; t++)
        {
            add  += dt_list[t].num_included(P);
        }

        post_dir_alpha = rep(1.0, P) + add;
        sample_dir_alpha(_, iter - 1) = post_dir_alpha;

        prop_prob = rdirichlet(1, post_dir_alpha);

        // making predictions
        for (int i = 0; i < m; i++)
        {
            dt_list[i].Predict_ord(Tree, Xcut, Xpred_test, n_test);
        }

        NumericVector predicted_mu = rowSums(Tree);
        sample_Z(_, iter - 1) = predicted_mu;
        

        // gathering posterior samples with thin, burn-in, n_post, thin_count
        if ((iter > burn_in) && (iter % thin == 0))
        {
            post_Z(_,post_iter) = sample_Z(_,iter-1);
           

            post_iter++;


        }



        Rcpp::checkUserInterrupt(); // check for break in R

    } // end of MCMC iterations
    compute_posterior_mean(post_Z, post_mean_predicted, n_test);
    


    List L = List::create(
        Named("z_test")             = Z_test,
        Named("sample_Z")           = sample_Z,
        Named("dir_alpha")          = sample_dir_alpha,

        Named("posterior_Z")        = post_Z,
        Named("test_prediction")    = post_mean_predicted
    );

    return L;
}