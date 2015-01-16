// (C) Copyright 2007, David M. Blei and John D. Lafferty

// This file is part of CTM-C.

// CTM-C is free software; you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your
// option) any later version.

// CTM-C is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
// for more details.

// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
// USA

/*************************************************************************
 *
 * llna.c
 *
 * estimation of an llna model by variational em
 *
 *************************************************************************/

#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <string.h>
#include <gsl/gsl_blas.h>
#include <assert.h>

#include "corpus.h"
#include "ctm.h"
#include "inference.h"
#include "gsl-wrappers.h"
#include "params.h"


extern llna_params PARAMS;

/*
 * e step
 *
 */


/*void expectation(corpus* corpus, llna_model* model, llna_ss* ss,
                 double* avg_niter, double* total_lhood,
                 gsl_matrix* corpus_lambda, gsl_matrix* corpus_nu,
                 gsl_matrix* corpus_phi_sum,
                 short reset_var, double* converged_pct)*/

void expectation(corpus* corpus_user, corpus* corpus_item, ratings* r_u, ratings* r_v,ratings* r, llna_model* model, llna_ss* ss,
                 double* avg_niter, double* total_lhood,
                 llna_corpus_var * c_var, gsl_vector * c_rmse,
                 short reset_var, double* converged_pct)
{
    int i=0,j=0;
    llna_var_param* var;
    doc Udoc;
    doc Vdoc;

    double Ulhood, Vlhood, Utotal,Vtotal;
    gsl_vector Ulambda, Vlambda,  Unu, Vnu;
    gsl_vector* Uphi_sum;
    gsl_vector* Vphi_sum;

    *avg_niter = 0.0;
    *converged_pct = 0;
    Uphi_sum = gsl_vector_alloc(model->k);
    Vphi_sum = gsl_vector_alloc(model->k);
    Utotal=0; Vtotal=0;
    //for (i = 0; i < corpus->ndocs; i++)
    for(j=0;j<3;j++)
    {
    	reset_llna_ss(ss);
		for (i = 0; i < corpus_user->ndocs; i++)
		{
			printf("userID: %5d ", i+1);
			Udoc = corpus_user->docs[i];

			var = new_llna_Uvar_param(Udoc.nterms, model->k);  //对变分参数var分配存储空间
			//if (reset_var)    //重置变分参数，因为不同文档，var不同，所以要重置
			if(0)
			{
				init_Uvar_unif(var, &Udoc, model); // 这里的参数doc有什么用？应该没实际用处，只不过标明是哪个文档的而已吧

				//init_var_unif(var, &Vdoc, model);
			}

			else  //不重置，就把第i个文档的变分参数传给 var ---------
			{
				Ulambda = gsl_matrix_row(c_var->Ucorpus_lambda, i).vector; //这个作为优化的初始值
				//Vlambda = gsl_matrix_row(c_var->Vcorpus_lambda, i).vector;  //这个没用，实际要用很多vlambda
				Unu = gsl_matrix_row(c_var->Ucorpus_nu, i).vector;
				//Vnu = gsl_matrix_row(c_var->Vcorpus_nu, i).vector;

				init_Uvar(var, &Udoc, model, &Ulambda, &Unu);
			}

			//lhood = 0.01;
			Ulhood = Uvar_inference(var, &Udoc, &r_u[i], model, c_var);  //---------很重要：更新变分参数, doc+model=>var-------
		    update_Uexpected_ss(var, &Udoc,&r_u[i],c_var, ss);   //更新的ss,用于M步的参数估计
			Utotal += Ulhood;
			printf("Ulhood %5.5e   niter %5d\n", Ulhood, var->niter);
			*avg_niter += var->niter;           //avg_niter=var_inference过程中的平均迭代次数
			*converged_pct += var->converged;

			gsl_matrix_set_row(c_var->Ucorpus_lambda, i, var->Ulambda);
			gsl_matrix_set_row(c_var->Ucorpus_nu, i, var->Unu);
			col_sum(var->Uphi, Uphi_sum);   // 这里的计算策略对吗？不应该是word_count加权累加吗？
			gsl_matrix_set_row(c_var->Ucorpus_phi_sum, i, Uphi_sum);  // 这里的c_var->Ucorpus_phi_sum在之后用到了吗？

			free_llna_Uvar_param(var);
		}

		for (i = 0; i < corpus_item->ndocs; i++)
		{
			printf("itemID: %5d ", i+1);
			Vdoc = corpus_item->docs[i];

			var = new_llna_Vvar_param(Vdoc.nterms, model->k);  //对变分参数var分配存储空间
			//if (reset_var)    //重置变分参数，因为不同文档，var不同，所以要重置
			if (0)    //重置变分参数，因为不同文档，var不同，所以要重置
			{
				init_Vvar_unif(var, &Vdoc, model); // 这里的参数doc有什么用？应该没实际用处，只不过标明是哪个文档的而已吧

				//init_var_unif(var, &Vdoc, model);
			}

			else  //不重置，就把第i个文档的变分参数传给 var ---------这里从来没用过？？/
			{
				//Ulambda = gsl_matrix_row(c_var->Ucorpus_lambda, i).vector; //这个没用，实际需要用的是很多ulambda
				Vlambda = gsl_matrix_row(c_var->Vcorpus_lambda, i).vector;  //这个作为优化的初始值

				//Unu = gsl_matrix_row(c_var->Ucorpus_nu, i).vector;
				Vnu = gsl_matrix_row(c_var->Vcorpus_nu, i).vector;


				init_Vvar(var, &Vdoc, model,&Vlambda, &Vnu);
			}

			//lhood = 0.01;
			Vlhood = Vvar_inference(var, &Vdoc, &r_v[i], model, c_var);  //---------很重要：更新变分参数, doc+model=>var-------
		    update_Vexpected_ss(var, &Vdoc, &r_v[i],c_var, ss);   //更新的ss,用于M步的参数估计
			Vtotal += Vlhood;
			printf("Ulhood %5.5e   niter %5d\n", Ulhood, var->niter);
			*avg_niter += var->niter;           //avg_niter=var_inference过程中的平均迭代次数
			*converged_pct += var->converged;

			gsl_matrix_set_row(c_var->Vcorpus_lambda, i, var->Vlambda);
			gsl_matrix_set_row(c_var->Vcorpus_nu, i, var->Vnu);
			col_sum(var->Vphi, Vphi_sum);
			gsl_matrix_set_row(c_var->Vcorpus_phi_sum, i, Vphi_sum);

			free_llna_Vvar_param(var);
		}

		double rmse;
		gsl_vector * predict_r;
		predict_r = gsl_vector_alloc(r->nratings);
	    predict_y(c_var->Ucorpus_lambda,c_var->Vcorpus_lambda, r, predict_r);
	    get_rmse(r, predict_r,&rmse);
	    gsl_vector_set(c_rmse,j,rmse);
	    printf("train:rmse=%lf\n",rmse);

    }



    gsl_vector_free(Uphi_sum);
    gsl_vector_free(Vphi_sum);
    *avg_niter = *avg_niter / corpus_user->ndocs; //这里是否修改，没太大意义
    *converged_pct = *converged_pct / corpus_user->ndocs; //这里意义也不大
    *total_lhood = Utotal+Vtotal;
}


/*
 * m step
 *
 */

void cov_shrinkage(gsl_matrix* mle, int n, gsl_matrix* result)
{
    int p = mle->size1, i;
    double temp = 0, alpha = 0, tau = 0, log_lambda_s = 0;
    gsl_vector
        *lambda_star = gsl_vector_calloc(p),
        t, u,
        *eigen_vals = gsl_vector_calloc(p),
        *s_eigen_vals = gsl_vector_calloc(p);
    gsl_matrix
        *d = gsl_matrix_calloc(p,p),
        *eigen_vects = gsl_matrix_calloc(p,p),
        *s_eigen_vects = gsl_matrix_calloc(p,p),
        *result1 = gsl_matrix_calloc(p,p);

    // get eigen decomposition

    sym_eigen(mle, eigen_vals, eigen_vects);
    for (i = 0; i < p; i++)
    {

        // compute shrunken eigenvalues

        temp = 0;
        alpha = 1.0/(n+p+1-2*i);
        vset(lambda_star, i, n * alpha * vget(eigen_vals, i));
    }

    // get diagonal mle and eigen decomposition

    t = gsl_matrix_diagonal(d).vector;
    u = gsl_matrix_diagonal(mle).vector;
    gsl_vector_memcpy(&t, &u);
    sym_eigen(d, s_eigen_vals, s_eigen_vects);

    // compute tau^2

    for (i = 0; i < p; i++)
        log_lambda_s += log(vget(s_eigen_vals, i));
    log_lambda_s = log_lambda_s/p;
    for (i = 0; i < p; i++)
        tau += pow(log(vget(lambda_star, i)) - log_lambda_s, 2)/(p + 4) - 2.0 / n;

    // shrink \lambda* towards the structured eigenvalues

    for (i = 0; i < p; i++)
        vset(lambda_star, i,
             exp((2.0/n)/((2.0/n) + tau) * log_lambda_s +
                 tau/((2.0/n) + tau) * log(vget(lambda_star, i))));

    // put the eigenvalues in a diagonal matrix

    t = gsl_matrix_diagonal(d).vector;
    gsl_vector_memcpy(&t, lambda_star);

    // reconstruct the covariance matrix

    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1, d, eigen_vects, 0, result1);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1, eigen_vects, result1, 0, result);

    // clean up

    gsl_vector_free(lambda_star);
    gsl_vector_free(eigen_vals);
    gsl_vector_free(s_eigen_vals);
    gsl_matrix_free(d);
    gsl_matrix_free(eigen_vects);
    gsl_matrix_free(s_eigen_vects);
    gsl_matrix_free(result1);
}



void maximization(llna_model* model, llna_ss * ss)
{
    int i, j;
    double sum;

    // 1. mean maximization  更新 model->mu




    for (i = 0; i < model->k; i++) {
    	vset(model->Umu, i, vget(ss->Umu_ss, i) / ss->Undata);
    	vset(model->Vmu, i, vget(ss->Vmu_ss, i) / ss->Vndata);
    }


    // 2. covariance maximization  更新 model->cov

    for (i = 0; i < model->k; i++)
    {
        for (j = 0; j < model->k; j++)
        {
            mset(model->Ucov, i, j,
                 (1.0 / ss->Undata) *
                 (mget(ss->Ucov_ss, i, j) +
                  ss->Undata * vget(model->Umu, i) * vget(model->Umu, j) -
                  vget(ss->Umu_ss, i) * vget(model->Umu, j) -
                  vget(ss->Umu_ss, j) * vget(model->Umu, i)));
        }
    }
    if (PARAMS.cov_estimate == SHRINK)
    {
        cov_shrinkage(model->Ucov, ss->Undata, model->Ucov);
    }
    matrix_inverse(model->Ucov, model->Uinv_cov);
    model->Ulog_det_inv_cov = log_det(model->Uinv_cov);

    for (i = 0; i < model->k; i++)
    {
        for (j = 0; j < model->k; j++)
        {
            mset(model->Vcov, i, j,
                 (1.0 / ss->Vndata) *
                 (mget(ss->Vcov_ss, i, j) +
                  ss->Vndata * vget(model->Vmu, i) * vget(model->Vmu, j) -
                  vget(ss->Vmu_ss, i) * vget(model->Vmu, j) -
                  vget(ss->Vmu_ss, j) * vget(model->Vmu, i)));
        }
    }
    if (PARAMS.cov_estimate == SHRINK)
    {
        cov_shrinkage(model->Vcov, ss->Vndata, model->Vcov);
    }
    matrix_inverse(model->Vcov, model->Vinv_cov);
    model->Vlog_det_inv_cov = log_det(model->Vinv_cov);

    // 3. topic maximization  更新 model->log_beta

    for (i = 0; i < model->k; i++)
    {
        sum = 0;
        for (j = 0; j < model->log_beta->size2; j++)
            sum += mget(ss->Ubeta_ss, i, j)+mget(ss->Vbeta_ss, i, j);


        if (sum == 0) sum = safe_log(sum) * model->log_beta->size2;
        else sum = safe_log(sum);

        for (j = 0; j < model->log_beta->size2; j++)
            mset(model->log_beta, i, j, safe_log(mget(ss->Ubeta_ss, i, j)+mget(ss->Vbeta_ss, i, j)) - sum);
    }


    // 5. update cov
    model->cov = ss->cov_ss / ss->nratings;  //可以让其为定值，不更新，因为有时竟然会更新得特别大




}


/*
 * run em
 *
 */
// 根据start的值<rand/seed/model> ，来初始化模型
llna_model* em_initial_model(int k, corpus* corpus_user, corpus* corpus_item, char* start)
{
    llna_model* model;
    printf("starting from %s\n", start);
    if (strcmp(start, "rand")==0)
        model = random_init(k, corpus_user->nterms, corpus_user->ndocs, corpus_item->ndocs);  //这是随机初始化，0均值，单位阵方差，beta为随机值
/*
    else if (strcmp(start, "seed")==0)
        model = corpus_init(k, corpus);    //这是固定种子随机初始化
*/
    else
        model = read_llna_model(start,corpus_user->ndocs, corpus_item->ndocs);
    return(model);
}


void em(char* dataset_user, char* dataset_item, char* dataset_rating, int k, char* start, char* dir)
{
    FILE* lhood_fptr;
    char string[100];
    int iteration = 0;
    double convergence = 1, lhood = 0, lhood_old = 0;
    corpus* corpus_user;
    corpus* corpus_item;
    //corpus* corpus;  //---
    ratings* r;
    ratings* r_u;
    ratings* r_v;
    llna_model *model;
    llna_ss* ss;  //这应该也需要更改
    time_t t1,t2;
    double avg_niter, converged_pct, old_conv = 0;

    llna_corpus_var * c_var;
    short reset_var = 1;

    // read the data and make the directory

    corpus_user = read_data(dataset_user);  //数据格式类似lda-c
    corpus_item = read_data(dataset_item);
    //corpus = read_data(dataset_item);  //--
    r = read_rating(dataset_rating); // 数据格式R×3
    r_u = get_uratings(r,corpus_user->ndocs);
    r_v = get_vratings(r,corpus_item->ndocs);

    mkdir(dir, S_IRUSR|S_IWUSR|S_IXUSR);

    // set up the log likelihood log file

    sprintf(string, "%s/likelihood.dat", dir);
    lhood_fptr = fopen(string, "w");

    // run em

    model = em_initial_model(k, corpus_user,corpus_item, start); //因为只采用random初始化，因此只用到了topic数目k
    ss = new_llna_ss(model);    //申请ss内部变量存储空间空间,也需要改---还未改----------------------

    //corpus_lambda = gsl_matrix_alloc(corpus->ndocs, model->k); //--
    c_var = new_llna_corpus_var(corpus_user->ndocs,corpus_item->ndocs,model->k);
    init_corpus_var(c_var, start);

    time(&t1);
    //init_temp_vectors(model->k-1); // !!! hacky 这是干嘛的？？
    init_temp_vectors(model->k);


    int iter_start;
    if (atoi(start) != NULL) {
    	iter_start = atoi(start);
    } else {
    	iter_start = 0;
    }

    // write start model & c_var & rmse_start
    sprintf(string, "%s/start-%03d", dir, iteration+iter_start);  //这是确认
    write_llna_model(model, string);  //已改
    write_c_var(c_var, string);

	double rmse_start;
	gsl_vector * predict_r;
	predict_r = gsl_vector_alloc(r->nratings);
    predict_y(c_var->Ucorpus_lambda,c_var->Vcorpus_lambda, r, predict_r);
    get_rmse(r, predict_r,&rmse_start);

    FILE* fileptr;
    char filename[200];
    sprintf(filename, "%s-rmse.dat", string);
    fileptr = fopen(filename, "w");
    fprintf(fileptr, "%lf\n", rmse_start);
    fclose(fileptr);



    gsl_vector * rmse;

    rmse = gsl_vector_alloc(3); //最好是4
    //===========这里是核心框架=======
    do
    {

        printf("***** EM ITERATION %d *****\n", iteration);

        //===========E-step=======
/*        expectation(corpus, model, ss, &avg_niter, &lhood,
                    corpus_lambda, corpus_nu, corpus_phi_sum,
                    reset_var, &converged_pct);*/
        expectation(corpus_user, corpus_item, r_u, r_v, r, model, ss, &avg_niter, &lhood,
                    c_var,rmse,
                    reset_var, &converged_pct);
        time(&t2);
        convergence = (lhood_old - lhood) / lhood_old;
        fprintf(lhood_fptr, "%d %5.5e %5.5e %5ld %5.5f %1.5f\n",
                iteration, lhood, convergence, (int) t2 - t1, avg_niter, converged_pct);

        if (((iteration % PARAMS.lag)==0) || isnan(lhood))
        {
            sprintf(string, "%s/%03d", dir, iteration + 1 + iter_start);
            write_llna_model(model, string);
            write_c_var(c_var, string);


            sprintf(string, "%s/%03d-rmse.dat", dir, iteration + 1 + iter_start);
            printf_vector(string, rmse);
        }
        time(&t1);

        printf("convergence is :%f\n",convergence);
        if (convergence < 0)  //如果收敛
        {
            reset_var = 0;  //这是干嘛的？--------------
            if (PARAMS.var_max_iter > 0)
                PARAMS.var_max_iter += 10;
            else PARAMS.var_convergence /= 10;
        }
        else  //如果还未收敛，继续进行m步，convergence迭代过程应该会变小
        {
        	//===========M-step=======
            maximization(model, ss);
            lhood_old = lhood;
            reset_var = 1;  //这是干嘛的
            iteration++;
        }

        fflush(lhood_fptr);


        old_conv = convergence;
    }
    while ((iteration < PARAMS.em_max_iter));
/*    while ((iteration < PARAMS.em_max_iter) &&
    		((convergence > 0.003) || (convergence < 0)));*/
           //((convergence > PARAMS.em_convergence) || (convergence < 0)));

    sprintf(string, "%s/final", dir);
    write_llna_model(model, string);
    write_c_var(c_var, string);

    sprintf(string, "%s/final-iteration.dat", dir);
    printf_value(string,iteration-1);


    fclose(lhood_fptr);
}


/*
 * load a model, and do approximate inference for each document in a corpus
 * inference过程中，corpus_user和item都不变，只不过是给定了一些新的user-item对，来预测其评分而已
 *
 */

void inference(char* dataset_user, char* dataset_item, char* dataset_rating, char* model_root, char* out)
{
    int i, nusers, nitems, iteration;
    char fname[100];
    double rmse;
    corpus* corpus_user;
    corpus* corpus_item;
    ratings* ratings;
    gsl_vector * predict_r;

    // read the data and model
    corpus_user = read_data(dataset_user);
    corpus_item = read_data(dataset_item);
    ratings = read_rating(dataset_rating);
    nusers=corpus_user->ndocs;
    nitems=corpus_item->ndocs;

    predict_r = gsl_vector_alloc(ratings->nratings);

    llna_model * model = read_llna_model("000",nusers,nitems);

    sprintf(fname, "%s-iteration.dat", model_root);
    scanf_value(fname, &iteration);

    gsl_matrix * Ucorpus_lambda = gsl_matrix_alloc(nusers, model->k);
    gsl_matrix * Vcorpus_lambda = gsl_matrix_alloc(nitems, model->k);

    for(i=0;i<=iteration+1;i++)
    {
        sprintf(fname, "%03d-Ucorpus_lambda.dat", i);
        scanf_matrix(fname, Ucorpus_lambda);

        sprintf(fname, "%03d-Vcorpus_lambda.dat", i);
        scanf_matrix(fname, Vcorpus_lambda);

        predict_y(Ucorpus_lambda,Vcorpus_lambda, ratings, predict_r);
        get_rmse(ratings, predict_r,&rmse);

        printf("iteration:%03d,inference:\trmse=%lf\n",i,rmse);

    }

    sprintf(fname, "%s-Ulambda.dat", model_root);
    scanf_matrix(fname, Ucorpus_lambda);

    sprintf(fname, "%s-Vlambda.dat", model_root);
    scanf_matrix(fname, Vcorpus_lambda);

    predict_y(Ucorpus_lambda,Vcorpus_lambda, ratings, predict_r);
    get_rmse(ratings, predict_r,&rmse);
    printf("%s,inference:\trmse=%lf\n",model_root,rmse);

}


/*
 * split documents into two random parts
 *
 */

void within_doc_split(char* dataset, char* src_data, char* dest_data, double prop)
{
    int i;
    corpus * corp, * dest_corp;

    corp = read_data(dataset);
    dest_corp = malloc(sizeof(corpus));
    printf("splitting %d docs\n", corp->ndocs);
    dest_corp->docs = malloc(sizeof(doc) * corp->ndocs);
    dest_corp->nterms = corp->nterms;
    dest_corp->ndocs = corp->ndocs;
    for (i = 0; i < corp->ndocs; i++)
        split(&(corp->docs[i]), &(dest_corp->docs[i]), prop);
    write_corpus(dest_corp, dest_data);
    write_corpus(corp, src_data);
}


/*
 * for each partially observed document: (a) perform inference on the
 * observations (b) take expected theta and compute likelihood
 *
 */

int pod_experiment(char* observed_data, char* heldout_data,
                   char* model_root, char* out)
{/*
    corpus *obs, *heldout;
    llna_model *model;
    llna_var_param *var;
    int i;
    gsl_vector *log_lhood, *e_theta;
    doc obs_doc, heldout_doc;
    char string[100];
    double total_lhood = 0, total_words = 0, l;
    FILE* e_theta_file = fopen("/Users/blei/llna050_e_theta.txt", "w");

    // load model and data
    obs = read_data(observed_data);
    heldout = read_data(heldout_data);
    assert(obs->ndocs == heldout->ndocs);
    model = read_llna_model(model_root);

    // run experiment
    init_temp_vectors(model->k-1); // !!! hacky
    log_lhood = gsl_vector_alloc(obs->ndocs + 1);
    e_theta = gsl_vector_alloc(model->k);
    for (i = 0; i < obs->ndocs; i++)
    {
        // get observed and heldout documents
        obs_doc = obs->docs[i];
        heldout_doc = heldout->docs[i];
        // compute variational distribution
        var = new_llna_var_param(obs_doc.nterms, model->k);
        init_var_unif(var, &obs_doc, model);
        var_inference(var, &obs_doc, model);
        expected_theta(var, &obs_doc, model, e_theta);

        vfprint(e_theta, e_theta_file);

        // approximate inference of held out data
        l = log_mult_prob(&heldout_doc, e_theta, model->log_beta);
        vset(log_lhood, i, l);
        total_words += heldout_doc.total;
        total_lhood += l;
        printf("hid doc %d    log_lhood %5.5f\n", i, vget(log_lhood, i));
        // save results?
        free_llna_var_param(var);
    }
    vset(log_lhood, obs->ndocs, exp(-total_lhood/total_words));
    printf("perplexity : %5.10f", exp(-total_lhood/total_words));
    sprintf(string, "%s-pod-llna.dat", out);
    printf_vector(string, log_lhood);*/
    return(0);
}


/*
 * little function to count the words in each document and spit it out
 *
 */

void count(char* corpus_name, char* output_name)
{
    corpus *c;
    int i;
    FILE *f;
    int j;
    f = fopen(output_name, "w");
    c = read_data(corpus_name);
    for (i = 0; i < c->ndocs; i++)
    {
        j = c->docs[i].total;
        fprintf(f, "%5d\n", j);
    }
}

/*
 * main function
 *
 */

int main(int argc, char* argv[])
{
    if (argc > 1)
    {
        if (strcmp(argv[1], "est")==0)
        {
            read_params(argv[8]);
            print_params();

            em(argv[2], argv[3], argv[4], atoi(argv[5]), argv[6], argv[7]);

            inference("ap_author.dat","ap_conjou.dat","ap_rating_test.dat","final","out");
            //inference("ap_author.dat","ap_pub.dat","ap_rating_test.dat","final","out");
            //inference("ap_user.dat","ap_movie.dat","ap_rating_test1.dat","final","out");
            //inference("ap_user.dat","ap_item.dat","ap_rating_test1.dat","final","out");

            return(0);
        }
        if (strcmp(argv[1], "inf")==0)
        {
            read_params(argv[7]);
            print_params();
            inference(argv[2], argv[3], argv[4],argv[5],argv[6]);
            return(0);
        }
    }
    printf("usage : ctm est <dataset_user> <dataset_item> <dataset_rating> <# topics> <rand/seed/model> <dir> <settings>\n");
    printf("        ctm inf <dataset_user> <dataset_item> <dataset_rating> <model-prefix> <results-prefix> <settings>\n");  //未改
    return(0);
}
