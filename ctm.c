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
 * reading, writing, and initializing a logistic normal allocation model
 *
 *************************************************************************/

#include <stdlib.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <time.h>

#include "gsl-wrappers.h"
#include "corpus.h"
#include "ctm.h"

/*
 * create a new empty model
 *
 */

llna_model* new_llna_model(int ntopics, int nterms, int nuser, int nitem)
{
    llna_model* model = malloc(sizeof(llna_model));
    model->k = ntopics;
    //model->mu = gsl_vector_calloc(ntopics - 1);
    model->Umu = gsl_vector_calloc(ntopics);  //注意大小设置，不知这样对不对？？是否要-1？？
    model->Vmu = gsl_vector_calloc(ntopics);

    model->Ucov = gsl_matrix_calloc(ntopics, ntopics);
    model->Vcov = gsl_matrix_calloc(ntopics, ntopics);


    model->Uinv_cov = gsl_matrix_calloc(ntopics, ntopics);
    model->Vinv_cov = gsl_matrix_calloc(ntopics, ntopics);

    model->log_beta = gsl_matrix_calloc(ntopics, nterms);  //为什么其他的要减一，这个不减一？？

    return(model);
}


/*
 * create and delete sufficient statistics
 *
 */

llna_ss * new_llna_ss(llna_model* model)
{
    llna_ss * ss;
    ss = malloc(sizeof(llna_ss));
    ss->Umu_ss = gsl_vector_calloc(model->k);
    ss->Vmu_ss = gsl_vector_calloc(model->k);

    ss->Ucov_ss = gsl_matrix_calloc(model->k, model->k);
    ss->Vcov_ss = gsl_matrix_calloc(model->k, model->k);

    ss->Ubeta_ss = gsl_matrix_calloc(model->k, model->log_beta->size2);
    ss->Vbeta_ss = gsl_matrix_calloc(model->k, model->log_beta->size2);
    ss->cov_ss = 0;
    ss->Undata = 0;
    ss->Vndata = 0;
    reset_llna_ss(ss);
    return(ss);
}


void del_llna_ss(llna_ss * ss)
{
    gsl_vector_free(ss->Umu_ss);
    gsl_vector_free(ss->Vmu_ss);
    gsl_matrix_free(ss->Ucov_ss);
    gsl_matrix_free(ss->Vcov_ss);
    gsl_matrix_free(ss->Ubeta_ss);
    gsl_matrix_free(ss->Vbeta_ss);
}

void reset_llna_ss(llna_ss * ss)
{
    gsl_matrix_set_all(ss->Ubeta_ss, 0);
    gsl_matrix_set_all(ss->Vbeta_ss, 0);
    gsl_matrix_set_all(ss->Ucov_ss, 0);
    gsl_matrix_set_all(ss->Vcov_ss, 0);
    gsl_vector_set_all(ss->Umu_ss, 0);
    gsl_vector_set_all(ss->Vmu_ss, 0);
    ss->cov_ss = 0;
    ss->Undata = 0;
    ss->Vndata = 0;
    ss->nratings = 0;
}


void write_ss(llna_ss * ss)
{
    printf_matrix("Ucov_ss", ss->Ucov_ss);
    printf_matrix("Vcov_ss", ss->Vcov_ss);
    printf_matrix("Ubeta_ss", ss->Ubeta_ss);
    printf_matrix("Vbeta_ss", ss->Vbeta_ss);
    printf_vector("Umu_ss", ss->Umu_ss);
    printf_vector("Vmu_ss", ss->Vmu_ss);
}
/*
 * initialize a model with zero-mean, diagonal covariance gaussian and
 * topics seeded from the corpus
 *
 */

llna_model* corpus_init(int ntopics, corpus* corpus)
{/*
    llna_model* model = new_llna_model(ntopics, corpus->nterms);
    gsl_rng * r = gsl_rng_alloc(gsl_rng_taus);
    doc* doc;
    int i, k, n, d;
    double sum;
    time_t seed;
    time(&seed);
    printf("SEED = %ld\n", seed);
    printf("USING 1115574245\n");
    gsl_rng_set(r, (long) 1115574245);
    // gsl_rng_set(r, (long) seed);
    // gsl_rng_set(r, (long) 432403824);

    // gaussian
    for (i = 0; i < ntopics-1; i++)
    {
        vset(model->mu, i, 0);
        mset(model->cov, i, i, 1.0);
    }
    matrix_inverse(model->cov, model->inv_cov);
    model->log_det_inv_cov = log_det(model->inv_cov);

    // topics
    for (k = 0; k < ntopics; k++)
    {
        sum = 0;
        // seed
        for (i = 0; i < NUM_INIT; i++)
        {
            d = floor(gsl_rng_uniform(r)*corpus->ndocs);
            printf("initialized with document %d\n", d);
            doc = &(corpus->docs[d]);
            for (n = 0; n < doc->nterms; n++)
            {
                minc(model->log_beta, k, doc->word[n], (double) doc->count[n]);
            }
        }
        // smooth
        for (n = 0; n < model->log_beta->size2; n++)
        {
            minc(model->log_beta, k, n, SEED_INIT_SMOOTH + gsl_rng_uniform(r));
            // minc(model->log_beta, k, n, SEED_INIT_SMOOTH);
            sum += mget(model->log_beta, k, n);
        }
        sum = safe_log(sum);
        // normalize
        for (n = 0; n < model->log_beta->size2; n++)
        {
            mset(model->log_beta, k, n,
                 safe_log(mget(model->log_beta, k, n)) - sum);
        }
    }
    gsl_rng_free(r);
    return(model);*/
	return(0);
}

/*
 * random initialization means zero-mean, diagonal covariance gaussian
 * and randomly generated topics
 *
 */

llna_model* random_init(int ntopics, int nterms, int nuser, int nitem)
{
    int i, j;
    double sum, val;
    llna_model* model = new_llna_model(ntopics, nterms, nuser, nitem);  //用于分配model内变量存储空间
    gsl_rng * r = gsl_rng_alloc(gsl_rng_taus);  //用陶斯沃特方法 ，定义一个随机数生成器r
    long t1;
    (void) time(&t1);
    // !!! DEBUG
    // t1 = gsl_rng_set(r, (long) 1115574245);
    printf("RANDOM SEED = %ld\n", t1);
    gsl_rng_set(r, t1);   //seed the random number generator,如果两次的种子相同，那么生成的值也相同


    model->cov = 1.0;   //评分y～N（U*V，cov）是一维的
    //model->inv_cov = 1.0;

    //初始化 mu,cov------均值mu设置为0,方差设置为单位阵
    for (i = 0; i < ntopics; i++)
    {
        vset(model->Umu, i, 0);
        vset(model->Vmu, i, 0);

        mset(model->Ucov, i, i, 1.0); //只是设置了对角元素，其余元素需要设置为0吗？
        mset(model->Vcov, i, i, 1.0);
    }


    //初始化log_beta
    for (i = 0; i < ntopics; i++)
    {
        sum = 0;
        for (j = 0; j < nterms; j++)
        {
            val = gsl_rng_uniform(r) + 1.0/100;  //gsl_rng_uniform生成[0,1)随机数,后者估计是用来平滑的
            sum += val;
            mset(model->log_beta, i, j, val);
        }
        for (j = 0; j < nterms; j++)
            mset(model->log_beta, i, j, log(mget(model->log_beta, i, j) / sum));
    }
    //matrix_inverse(model->cov, model->inv_cov);
    //printf("%lf,%f\n",model->cov->data[0],model->inv_cov->data[0]);

    //model->log_det_inv_cov = ntopics*model->inv_cov;
    matrix_inverse(model->Ucov, model->Uinv_cov);
    model->Ulog_det_inv_cov = log_det(model->Uinv_cov);
    matrix_inverse(model->Vcov, model->Vinv_cov);
    model->Vlog_det_inv_cov = log_det(model->Vinv_cov);

    //model->log_det_inv_cov = 0.0 ;


    //printf("%lf",model->log_det_inv_cov);  //log(1)=0
    gsl_rng_free(r);
    return(model);
}

void predict_y(gsl_matrix * Ucorpus_lambda,gsl_matrix * Vcorpus_lambda, ratings* r, gsl_vector * predict_r)
{
	int i,j,user,item;
	double y;

	for(i=0;i<r->nratings;i++)
	{
		y=0.0;
		user=r->users[i];
		item=r->items[i];
		for(j=0;j<Ucorpus_lambda->size2;j++)
		{
			y += mget(Ucorpus_lambda,user-1,j) * mget(Vcorpus_lambda,item-1,j);
		}


		vset(predict_r,i,y);
	}

}
void get_rmse(ratings * r, gsl_vector * predict_r, double * rmse)
{
	int i;
	double err,y_clip;
	*rmse = 0.0;
	for(i=0;i<r->nratings;i++)
	{
		y_clip = vget(predict_r,i);
		if(y_clip > 5)
			y_clip = 5;
		else if(y_clip < 1)
			y_clip = 1;
		err=r->ratings[i]-y_clip;
		*rmse += err*err;

	}
	*rmse = sqrt(*rmse/r->nratings);

}

/*
 * read a model
 *
 */

llna_model* read_llna_model(char * root,int nusers, int nitems)
{
    char filename[200];
    FILE* fileptr;
    llna_model* model;
    int ntopics, nterms;

    // 1. read parameters and allocate model
    sprintf(filename, "%s-param.txt", root);
    printf("reading params from %s\n", filename);
    fileptr = fopen(filename, "r");
    fscanf(fileptr, "num_topics %d\n", &ntopics);
    fscanf(fileptr, "num_terms %d\n", &nterms);
    fclose(fileptr);
    printf("%d topics, %d terms\n", ntopics, nterms);
    // allocate model
    model = new_llna_model(ntopics, nterms, nusers, nitems);




    // 2. read model


    // read gaussian
    printf("reading gaussian\n");
    sprintf(filename, "%s-Umu.dat", root);
    scanf_vector(filename, model->Umu);
    sprintf(filename, "%s-Vmu.dat", root);
    scanf_vector(filename, model->Vmu);

    sprintf(filename, "%s-cov.dat", root);
    fileptr = fopen(filename, "r");
    fscanf(fileptr, "%lf\n", &(model->cov));
    fclose(fileptr);

    sprintf(filename, "%s-Ucov.dat", root);
    scanf_matrix(filename, model->Ucov);
    sprintf(filename, "%s-Vcov.dat", root);
    scanf_matrix(filename, model->Vcov);

    sprintf(filename, "%s-Uinv-cov.dat", root);
    scanf_matrix(filename, model->Uinv_cov);
    sprintf(filename, "%s-Vinv-cov.dat", root);
    scanf_matrix(filename, model->Vinv_cov);

    sprintf(filename, "%s-Ulog-det-inv-cov.dat", root);
    fileptr = fopen(filename, "r");
    fscanf(fileptr, "%lf\n", &(model->Ulog_det_inv_cov));
    fclose(fileptr);
    sprintf(filename, "%s-Vlog-det-inv-cov.dat", root);
    fileptr = fopen(filename, "r");
    fscanf(fileptr, "%lf\n", &(model->Vlog_det_inv_cov));
    fclose(fileptr);

    // read topic matrix
    printf("reading topics\n");
    sprintf(filename, "%s-log-beta.dat", root);
    scanf_matrix(filename, model->log_beta);



    return(model);

}

/*
 * write a model
 *
 */

void write_llna_model(llna_model * model, char * root)  //已经改完
{
    char filename[200];
    FILE* fileptr;

    // write parameters
    printf("writing params\n");
    sprintf(filename, "%s-param.txt", root);
    fileptr = fopen(filename, "w");
    fprintf(fileptr, "num_topics %d\n", model->k);
    fprintf(fileptr, "num_terms %d\n", (int) model->log_beta->size2);

    fclose(fileptr);
    // write gaussian
    printf("writing gaussian\n");

    sprintf(filename, "%s-Umu.dat", root);
    printf_vector(filename, model->Umu);
    sprintf(filename, "%s-Vmu.dat", root);
    printf_vector(filename, model->Vmu);

    //sprintf(filename, "%s-cov.dat", root);
    //printf_matrix(filename, model->cov);
    sprintf(filename, "%s-cov.dat", root);
    fileptr = fopen(filename, "w");
    fprintf(fileptr, "%lf\n", model->cov);
    fclose(fileptr);

    sprintf(filename, "%s-Ucov.dat", root);
    printf_matrix(filename, model->Ucov);
    sprintf(filename, "%s-Vcov.dat", root);
    printf_matrix(filename, model->Vcov);



    //sprintf(filename, "%s-inv-cov.dat", root);
    //sprintf(filename, "%s-inv-cov.dat", root);
    //fileptr = fopen(filename, "w");
    //fprintf(fileptr, "%lf\n", model->inv_cov);
    //fclose(fileptr);


    sprintf(filename, "%s-Uinv-cov.dat", root);
    printf_matrix(filename, model->Uinv_cov);
    sprintf(filename, "%s-Vinv-cov.dat", root);
    printf_matrix(filename, model->Vinv_cov);

    //sprintf(filename, "%s-log-det-inv-cov.dat", root);
    //fileptr = fopen(filename, "w");
    //fprintf(fileptr, "%lf\n", model->log_det_inv_cov);
    //fclose(fileptr);
    sprintf(filename, "%s-Ulog-det-inv-cov.dat", root);
    fileptr = fopen(filename, "w");
    fprintf(fileptr, "%lf\n", model->Ulog_det_inv_cov);
    fclose(fileptr);
    sprintf(filename, "%s-Vlog-det-inv-cov.dat", root);
    fileptr = fopen(filename, "w");
    fprintf(fileptr, "%lf\n", model->Vlog_det_inv_cov);
    fclose(fileptr);

    // write topic matrix
    printf("writing topics\n");
    sprintf(filename, "%s-log-beta.dat", root);
    printf_matrix(filename, model->log_beta);
}


