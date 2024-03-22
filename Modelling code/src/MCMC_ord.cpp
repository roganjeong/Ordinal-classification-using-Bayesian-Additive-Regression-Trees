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
    const int n_iter, // "int delta_default = 0" is gone temporarily // "const bool proportional_initialzation = true" is gone temporarily
    const int burn_in,
    const int thinning,
    const bool verbose = false
) {

    if (burn_in >= n_iter)
    {
        Rcout << "\'burn_in\' must be strictly less than \'n_iter\'" << std::endl;
        exit(-1);
    }

    if (thinning <= 0)
    {
        Rcout << "\'thinning\' must be strictly greater than \'0\'" << std::endl;
        exit(-1);
    }
    
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
    NumericVector alphas = rep(0.0, K-1);
    initial_alphas(alphas, Y, K);

    NumericVector Z = rep(0.0, n);
    initial_Z(Z, Y, K, alphas);// Rcpp::rnorm(n, 0, 1); // latent variable - original :  Z = Rcpp::rnorm(n, R::qnorm(mean(Y_trt), 0, 1, true, false), 1)

    NumericVector prob = {p_grow, p_prune, p_change};

    double sigma2 = 1.0;
    //NumericVector dir_alpha_hist (n_iter + 1); // create placeholder for dir_alpha
    //dir_alpha_hist(0) = dir_alpha;

    // sigma_mu based on min/max of Y
    const double sigma_mu   = (Rcpp::max(Z) - Rcpp::min(Z))/(4*sqrt(m)); // original : std::max(pow(min(Z)     / (-2 * sqrt(m)), 2), pow(max(Z)     / (2 * sqrt(m)), 2))
    // 4 = 2*2 = 2*k, k is fixed as 2

    // Initial values of R
    NumericVector R  = clone(Z);

    // Initial values for the selection probabilities
    NumericVector post_dir_alpha  = rep(1.0, P);

    // thin = 10, burn-ins = n_iter/2
    int thin       = thinning;
    int burnt      = burn_in;
    int n_post     = (n_iter - burnt) / thin; // number of post sample
    int post_iter  = 0;
    // int thin_count = 1;

    // NumericMatrix rowSum_mu(n, n_iter);          // added           // for saving all the mu_i's of the n observations, for all iterations
    // NumericMatrix sample_Z(n, n_iter);                      //****** for saving all the Z_i's of the n observations, for all iterations
    // NumericMatrix sample_LBs(K-1, n_iter);                  // for saving all the upper bounds of the n observations, for all iterations
    // NumericMatrix sample_UBs(K-1, n_iter);                  // for saving all the lower bounds of the n observations, for all iterations
    // NumericMatrix sample_alphas(K-1, n_iter);               // for saving all of the K-1 alphas, for all iterations
    // NumericMatrix sample_dir_alpha(P, n_iter);              // for saving all the dirichlet alpha parameters of the P predictors, for all iterations
    // NumericMatrix sample_predicted(n_test, n_iter);         // for saving all the Z_i's of the observations of the test set, for all iterations
    // NumericMatrix predicted_rowSum_mu(n_test, n_iter); // added     // for saving all the mu_i's of the observations of the test set, for all iterations
    // NumericMatrix predicted_Z(n_test, n_iter);              // for saving all the Z_i's of the observations of the test set, for all iterations
    

    // NumericMatrix post_Z(n, n_post);                        //****** for saving all the Z_i's of the n observations, for n_post
    NumericMatrix post_Z_test(n_test, n_post);
    NumericMatrix post_LBs(K-1, n_post);                    // for saving all the upper bounds of the n observations, for n_post
    NumericMatrix post_UBs(K-1, n_post);                    // for saving all the lower bounds of the n observations, for n_post
    NumericMatrix post_alphas(K-1, n_post);                 // for saving all of the K-1 alphas, for n_post
    NumericMatrix post_predicted_mu(n_test, n_post);  // added      // for saving all the mu_i's of the observations of the test set, for n_post
    NumericMatrix post_predicted(n_test, n_post);           // for saving all the Z_i's of the observations of the test set, for n_post
    NumericMatrix probability_categories(n_test,K);        // print the probabilities of different categories for each test observation

    NumericMatrix post_mean_LBs(K-1, n_post);        //******
    NumericMatrix post_mean_UBs(K-1, n_post);        //******
    NumericMatrix post_mean_alphas(K-1, n_post);        //******
    NumericVector post_mode_predicted(n_test);

    // NumericMatrix post_probability()

    // post_Z(_, 0) = Z;
    // NumericVector predicted_Y_train(n);
    NumericVector predicted_Y = rep(0.0, n_test); // must be changed to 'IntegerVector' - now it should be changed to NumericVector
    // IntegerMatrix ind    (n_post, P);    // original : (n_post, P+1)

    IntegerMatrix Obs_list(n, m); // changed list to matrix

    // Place-holder for the posterior samples
    NumericMatrix Tree   (n, m);
    NumericMatrix Tree_pred;

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
    // int delta;
    // if (delta_default == 0)
    // {
    //     delta = 6 * P; // (6 / (K / 2)) * P
    // }
    // else
    // {
    //     delta = delta_default; // should delta never be updated in MCMC...? "abs(Rcpp::mean(mu)) + 3.0" --> "3.0"
    // }
    // now, the tree is empty so Rcpp::mean(mu) prints all zeros
    // const double negative_delta = delta*(-1);
    NumericVector LBs = rep(0.0, K-1);
    NumericVector UBs = rep(0.0, K-1);
    // NumericVector alphas = rep(0.0, K-1);

    // Rcout << "print mu vector : [" << mu << "]" << std::endl;

    // for (int u = 0; u < K-1; u++)
    // {
    //     if (u == 0)
    //     {
    //         alphas(u) = Rcpp::runif(1, negative_delta, delta)(0);
    //     }
    //     else
    //     {
    //         double alpha_past1 = alphas(u-1);
    //         alphas(u) = Rcpp::runif(1, alpha_past1, alpha_past1 + delta)(0);
    //     }
    // }
    // post_alphas(_,0) = alphas;
    
    // need to find a way to set more informative initial values for alphas

    ////////////////////////////////////////
    //////////   Run main MCMC    //////////
    ////////////////////////////////////////

    // Run MCMC
    NumericVector predicted_mu(n_test);
    for (int iter = 1; iter <= n_iter; iter++)
    {

        if (verbose)
        {
            if (iter == 1)
            {
                Rcout << "_______________________________________________________________________________" << std::endl;
                Rcout << "Rcpp iter : " << iter << " of " << n_iter << std::endl;
                Rcout << "alphas = (" << alphas << ")" << std::endl;
                Rcout << "_______________________________________________________________________________" << std::endl;
                // for (int t = 0; t < m; t++)
                // {
                //     // decision trees
                //     update_R(R, Z, Tree, t);

                //     if (dt_list[t].length() == 1)
                //     { 
                //         // tree has no node yet
                //         // grow first step
                //         dt_list[t].GROW_first(
                //             Xpred, Xcut, sigma2, sigma_mu, R, Obs_list,
                //             p_prune, p_grow, alpha, beta, prop_prob
                //         );
                //     }
                // }
            }
            if (iter % 100 == 0)
            {
                Rcout << "_______________________________________________________________________________" << std::endl;
                Rcout << "Rcpp iter : " << iter << " of " << n_iter << std::endl;
                Rcout << "Lower Bounds = (" << LBs << ")" << std::endl;
                Rcout << "Upper Bounds = (" << UBs << ")" << std::endl;
                Rcout << "alphas = (" << alphas << ")" << std::endl;
                Rcout << "_______________________________________________________________________________" << std::endl;
            }
        }

        // // check for nan values for Z
        // check_nan_Z(Z, n);

        // // check for nan values for LBs
        // check_nan_LBs(LBs, K);
    
        // // check for nan values for UBs
        // check_nan_UBs(UBs, K);

        // // check for nan values for alphas
        // check_nan_alphas(alphas, K);
        // if ((isnan(alphas(0)) == true) || (isnan(alphas(1)) == true) || (isnan(alphas(2)) == true))
        // {
        //     Rcout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
        //     Rcout << "nan occured at the iteration = " << iter << std::endl;
        //     Rcout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
        //     break;
        // }
        
        // initial_Z(Z, Y, K, alphas);
        update_Z(Z, Y, K, alphas, Tree); // original : update_Z(Z, Y_trt, Tree)
        // sample_Z(_, iter - 1) = Z;
        // Rcout << "update_Z : iter = " << iter << " out of " << n_iter << std::endl;
        // max_Z = Rcpp::max(Z);
        // med_Z = Rcpp::median(Z);
        // min_Z = Rcpp::min(Z);

        // sample_OrderStats(0, iter-1) = min_Z;
        // sample_OrderStats(1, iter-1) = med_Z;
        // sample_OrderStats(2, iter-1) = max_Z;

        // sample_sigma_mu(iter - 1) = sigma_mu;

        // sigma_mu   = (max_Z - min_Z)/(4*sqrt(m));
        // sample_sigma_mu(iter - 1) = sigma_mu;

        update_LBs(LBs, Z, Y, K, n, alphas); // update_LBs(LBs, Z, Y, K, n, alphas, delta);
        // sample_LBs(_, iter - 1) = LBs;
        // Rcout << "post_LBs : iter = " << iter << " out of " << n_iter << std::endl;

        update_UBs(UBs, Z, Y, K, n, alphas); //update_UBs(UBs, Z, Y, K, n, alphas, delta)
        // sample_UBs(_, iter - 1) = UBs;
        // Rcout << "post_UBs : iter = " << iter << " out of " << n_iter << std::endl;

        update_alphas(alphas, K, UBs, LBs);
        // sample_alphas(_, iter - 1) = alphas;
        // Rcout << "post_alphas : iter = " << iter << " out of " << n_iter << std::endl;

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
        // rowSum_mu(_,iter-1) = rowSums(Tree);

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
        // sample_dir_alpha(_, iter - 1) = post_dir_alpha;

        prop_prob = rdirichlet(1, post_dir_alpha);


        // Tree_pred = clone(Tree);
        // for (int i = 0; i < m; i++)
        // {
        //     dt_list[i].Predict_ord(Tree_pred, Xcut, Xpred_test, n_test);
        // }
        // predicted_mu = rowSums(Tree_pred);

        // for (int i = 0; i < n_test; i++)
        // {
        //     predicted_Z(i, iter-1) = predicted_mu(i);
        //     if (predicted_mu(i) >= alphas(K-2)) // alphas(0) == alpha_1, alphas(1) == alpha_2, alphas(2) == alpha_3
        //     {
        //         predicted_Y(i) = K; // predicted_Y(i) = post_predicted(n_test, n_post)
        //         sample_predicted(i, iter-1) = K;
        //     }
        //     else if (predicted_mu(i) < alphas(0))
        //     {
        //         predicted_Y(i) = 1;
        //         sample_predicted(i, iter-1) = 1;
        //     }
        //     else
        //     {
        //         for (int pred_ind = 1; pred_ind < K; pred_ind++) // pred_ind = 1, 2, 3
        //         {
        //             if ((predicted_mu(i) >= alphas(pred_ind - 1)) && (predicted_mu(i) < alphas(pred_ind)))
        //             {
        //                 predicted_Y(i) = pred_ind + 1;
        //                 sample_predicted(i, iter-1) = pred_ind + 1;
        //             }
        //         }
        //     }
        // }
        
        // how should I compute the cutoff points for the predictions
        // let's first try to omit the "compute_posterior_mean" process
        if ((iter > burnt) && (iter % thin == 0))
        {

            Tree_pred = clone(Tree);

            // making predictions : for making inferences form the test set
            for (int i = 0; i < m; i++)
            {
                dt_list[i].Predict_ord(Tree_pred, Xcut, Xpred_test, n_test);
            }
            predicted_mu = rowSums(Tree_pred);

            for (int i = 0; i < n_test; i++)
            {

                if (predicted_mu(i) >= alphas(K-2)) // alphas(0) == alpha_1, alphas(1) == alpha_2, alphas(2) == alpha_3
                {
                    predicted_Y(i) = K; // predicted_Y(i) = post_predicted(n_test, n_post)
                    // sample_predicted(i, iter-1) = K;
                    post_predicted(i, post_iter) = K;
                }
                else if (predicted_mu(i) < alphas(0))
                {
                    predicted_Y(i) = 1;
                    post_predicted(i, post_iter) = 1;
                }
                else
                {
                    for (int pred_ind = 1; pred_ind < K; pred_ind++) // pred_ind = 1, 2, 3
                    {
                        if ((predicted_mu(i) >= alphas(pred_ind - 1)) && (predicted_mu(i) < alphas(pred_ind)))
                        {
                            predicted_Y(i) = pred_ind + 1;
                            post_predicted(i, post_iter) = pred_ind + 1;
                        }
                    }
                }
            }




            // post_rowSum_mu(_, post_iter) = rowSum_mu(_,iter-1);
            // post_Z(_,post_iter) = sample_Z(_,iter-1);
            // post_Z_test(_,post_iter) = predicted_Z(_,iter-1);
            post_Z_test(_,post_iter) = predicted_mu; // highly likely that it is unnecessary
            
            // post_OrderStats(_,post_iter) = sample_OrderStats(_,iter-1);

            // post_LBs(_,post_iter) = sample_LBs(_,iter-1);
            post_LBs(_,post_iter) = LBs;

            // post_UBs(_,post_iter) = sample_UBs(_,iter-1);
            post_UBs(_,post_iter) = UBs;

            // post_alphas(_,post_iter) = sample_alphas(_,iter-1);
            post_alphas(_,post_iter) = alphas;

            // post_predicted_mu(_,post_iter) = predicted_rowSum_mu(_, iter-1);
            // post_predicted(_,post_iter) = sample_predicted(_,iter-1);
            post_predicted(_,post_iter) = predicted_Y;

            post_iter++;


        }

        // computing posterior mean for Z, LBs, UBs, alphas
        // compute_posterior_mean(post_LBs, post_mean_LBs, post_UBs, post_mean_UBs, post_alphas, post_mean_alphas)

        // for (int i = 0; i < n_test; i++)
        // {
        //     if (predicted_mu(i) >= post_mean_alphas(K-2)) // alphas(0) == alpha_1, alphas(1) == alpha_2, alphas(2) == alpha_3
        //     {
        //         predicted_Y(i) = K;
        //     }
        //     else if (predicted_mu(i) < post_mean_alphas(0))
        //     {
        //         predicted_Y(i) = 1;
        //     }
        //     else
        //     {
        //         for (int pred_ind = 1; pred_ind < K; pred_ind++) // pred_ind = 1, 2, 3
        //         {
        //             if ((predicted_mu(i) >= post_mean_alphas(pred_ind - 1)) && (predicted_mu(i) < post_mean_alphas(pred_ind)))
        //             {
        //                 predicted_Y(i) = pred_ind + 1;
        //             }
        //         }
        //     }
        // }


        Rcpp::checkUserInterrupt(); // check for break in R

    } // end of MCMC iterations
    
    // make predictions for test set by majority rule
    // post_predicted(i, post_iter)
    // for (int i = 0; i < n_test ; i++)
    // {
    //     find_mode(predicted_Y(i), post_predicted(i,_), K, n_post);
    // }

    eachcat_probability(probability_categories, post_predicted, K, n_test, n_post);
    
    find_mode(post_mode_predicted, post_predicted, K, n_test, n_post);


    // making predictions for training set
    // for (int i = 0; i < m; i++)
    // {
    //     dt_list[i].Predict_ord(Tree, Xcut, Xpred_test, n_test);
    // }
    // NumericVector predicted_mu = rowSums(Tree);

    // for (int i = 0; i < n_test; i++)
    // {
    //     if (predicted_mu(i) >= post_mean_alphas(K-2)) // alphas(0) == alpha_1, alphas(1) == alpha_2, alphas(2) == alpha_3
    //     {
    //         predicted_Y(i) = K;
    //     }
    //     else if (predicted_mu(i) < post_mean_alphas(0))
    //     {
    //         predicted_Y(i) = 1;
    //     }
    //     else
    //     {
    //         for (int pred_ind = 1; pred_ind < K; pred_ind++) // pred_ind = 1, 2, 3
    //         {
    //             if ((predicted_mu(i) >= post_mean_alphas(pred_ind - 1)) && (predicted_mu(i) < post_mean_alphas(pred_ind)))
    //             {
    //                 predicted_Y(i) = pred_ind + 1;
    //             }
    //         }
    //     }
    // }
    List L = List::create(
        Named("Y_test")                = Y_test,
        
        Named("posterior_LBs")         = post_LBs,
        Named("posterior_UBs")         = post_UBs,
        Named("posterior_Gammas")      = post_alphas,
        Named("posterior_predicted")   = post_predicted,

        Named("posterior_Z_test")      = post_Z_test,
        Named("prob_each_cat")         = probability_categories,
        Named("test_prediction")       = post_mode_predicted
    );

    return L;

}