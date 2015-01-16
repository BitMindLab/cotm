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

#include <stdlib.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_vector_double.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <assert.h>

#include "gsl-wrappers.h"
#include "corpus.h"
#include "ctm.h"
#include "params.h"
#include "inference.h"

extern llna_params PARAMS;

double f_Ulambda(const gsl_vector * p, void * params);
double f_Vlambda(const gsl_vector * p, void * params);
void df_Ulambda(const gsl_vector * p, void * params, gsl_vector * df);
void df_Vlambda(const gsl_vector * p, void * params, gsl_vector * df);
void fdf_Ulambda(const gsl_vector * p, void * params, double * f, gsl_vector * df);
void fdf_Vlambda(const gsl_vector * p, void * params, double * f, gsl_vector * df);

double f_nu(const gsl_vector * p, void * params);
void df_nu(const gsl_vector * p, void * params, gsl_vector * df);
void fdf_nu(const gsl_vector * p, void * params, double * f, gsl_vector * df);

/*
 * temporary k-1 vectors so we don't have to allocate, deallocate
 *
 */

gsl_vector ** temp;
int ntemp = 5;

void init_temp_vectors(int size)
{
    int i;
    temp = malloc(sizeof(gsl_vector *)*ntemp);
    for (i = 0; i < ntemp; i++)
        temp[i] = gsl_vector_alloc(size);
}


/*
 * likelihood bound
 *
 */

double Uexpect_mult_norm(llna_var_param * var)
{
    int i;
    double sum_exp = 0;
    int niter = var->Ulambda->size;

    for (i = 0; i < niter; i++)
        sum_exp += exp(vget(var->Ulambda, i) + (0.5) * vget(var->Unu,i));

    return((1.0/var->Uzeta) * sum_exp - 1.0 + log(var->Uzeta));
}
double Vexpect_mult_norm(llna_var_param * var)
{
    int i;
    double sum_exp = 0;
    int niter = var->Vlambda->size;

    for (i = 0; i < niter; i++)
        sum_exp += exp(vget(var->Vlambda, i) + (0.5) * vget(var->Vnu,i));

    return((1.0/var->Vzeta) * sum_exp - 1.0 + log(var->Vzeta));
}


void Ulhood_bnd(llna_var_param* var, doc* Udoc, ratings * r_ui, llna_model* mod, llna_corpus_var * c_var)
{
    int i = 0, j = 0, k = mod->k;
    gsl_vector_set_zero(var->Utopic_scores);
    //gsl_vector_set_zero(var->Vtopic_scores);

    // 1. E[log p(\eta | \mu, \Sigma)] + H(q(\eta | \lambda, \nu)
    // 1. U

    double lhood  = (0.5) * mod->Ulog_det_inv_cov + (0.5) * (mod->k);
    for (i = 0; i < k; i++)
    {
        double v = - (0.5) * vget(var->Unu, i) * mget(mod->Uinv_cov,i, i);
        for (j = 0; j < mod->k; j++)
        {
            v -= (0.5) *
                (vget(var->Ulambda, i) - vget(mod->Umu, i)) *
                mget(mod->Uinv_cov, i, j) *
                (vget(var->Ulambda, j) - vget(mod->Umu, j));
        }
        v += (0.5) * log(vget(var->Unu, i));
        lhood += v;
    }
    // 2.E[log p(y|eta1, eta2)]

    int item; double rating;
	gsl_vector Vlambda, Vnu;
	double tt,uv,t1,t2,t3;
	for (i = 0; i<r_ui->nratings; i++ )
	{
		item = r_ui->items[i];
		rating = r_ui->ratings[i];
		Vlambda = gsl_matrix_row(c_var->Vcorpus_lambda, item-1).vector;
		Vnu = gsl_matrix_row(c_var->Vcorpus_nu, item-1).vector;


		lhood += -0.5*log(mod->cov) - 0.5/mod->cov*rating *rating;
		gsl_blas_ddot(var->Ulambda,&Vlambda,&uv); //uu= Ulambda*Vlambda
		//gsl_blas_ddot(var->Ulambda,var->Vlambda,&t2);

		gsl_blas_dcopy(var->Ulambda, temp[0]);    // temp[0] = Ulambda^2
		gsl_blas_dcopy(&Vlambda, temp[1]);    // temp[1] = Vlambda^2
		for (j = 0; j < mod->k; j++) {
			tt = gsl_vector_get(temp[0],j)*gsl_vector_get(temp[0],j);
			gsl_vector_set(temp[0],j,tt);
			tt = gsl_vector_get(temp[1],j)*gsl_vector_get(temp[1],j);
			gsl_vector_set(temp[1],j,tt);
		}
		gsl_blas_ddot(temp[0],&Vnu,&t1);
		gsl_blas_ddot(temp[1],var->Unu,&t2);
		gsl_blas_ddot(var->Unu,&Vnu,&t3);

		lhood += -0.5/mod->cov*(uv*uv+t1+t2+t3)+rating/mod->cov*uv;
	}


    // 3.E[log p(z_n | \eta)] + E[log p(w_n | \beta)] + H(q(z_n | \phi_n))
    // 3.1 U
    lhood -= Uexpect_mult_norm(var) * Udoc->total;
    for (i = 0; i < Udoc->nterms; i++)
    {
        // !!! we can speed this up by turning it into a dot product
        // !!! profiler says this is where some time is spent
        for (j = 0; j < mod->k; j++)
        {
            double phi_ij = mget(var->Uphi, i, j);
            double log_phi_ij = mget(var->Ulog_phi, i, j);
            if (phi_ij > 0)
            {
                vinc(var->Utopic_scores, j, phi_ij * Udoc->count[i]);
                lhood +=
                    Udoc->count[i] * phi_ij *
                    (vget(var->Ulambda, j) +
                     mget(mod->log_beta, j, Udoc->word[i]) -
                     log_phi_ij);
            }
        }
    }


    var->lhood = lhood;

    //assert(!isnan(var->lhood));
}

void Vlhood_bnd(llna_var_param* var, doc* Vdoc, ratings * r_vj, llna_model* mod, llna_corpus_var * c_var)
{
    int i = 0, j = 0, k = mod->k;
    gsl_vector_set_zero(var->Vtopic_scores);
    //gsl_vector_set_zero(var->Vtopic_scores);

    // 1. E[log p(\eta | \mu, \Sigma)] + H(q(\eta | \lambda, \nu)
    // 1. V

    double lhood  = (0.5) * mod->Vlog_det_inv_cov + (0.5) * (mod->k);
    for (i = 0; i < k; i++)
    {
        double v = - (0.5) * vget(var->Vnu, i) * mget(mod->Vinv_cov,i, i);
        for (j = 0; j < mod->k; j++)
        {
            v -= (0.5) *
                (vget(var->Vlambda, i) - vget(mod->Vmu, i)) *
                mget(mod->Vinv_cov, i, j) *
                (vget(var->Vlambda, j) - vget(mod->Vmu, j));
        }
        v += (0.5) * log(vget(var->Vnu, i));
        lhood += v;
    }

    // 2.E[log p(y|eta1, eta2)]

    int user; double rating;
	gsl_vector Ulambda, Unu;
	double tt,uv,t1,t2,t3;
	for (i = 0; i<r_vj->nratings; i++ )
	{
		user = r_vj->users[i];
		rating = r_vj->ratings[i];
		Ulambda = gsl_matrix_row(c_var->Ucorpus_lambda, user-1).vector;
		Unu = gsl_matrix_row(c_var->Ucorpus_nu, user-1).vector;


		lhood += -0.5*log(mod->cov) - 0.5/mod->cov*rating *rating;
		gsl_blas_ddot(var->Vlambda,&Ulambda,&uv); //uu= Ulambda*Vlambda
		//gsl_blas_ddot(var->Ulambda,var->Vlambda,&t2);

		gsl_blas_dcopy(var->Vlambda, temp[0]);    // temp[0] = Ulambda^2
		gsl_blas_dcopy(&Ulambda, temp[1]);    // temp[1] = Vlambda^2
		for (j = 0; j < mod->k; j++) {
			tt = gsl_vector_get(temp[0],j)*gsl_vector_get(temp[0],j);
			gsl_vector_set(temp[0],j,tt);
			tt = gsl_vector_get(temp[1],j)*gsl_vector_get(temp[1],j);
			gsl_vector_set(temp[1],j,tt);
		}
		gsl_blas_ddot(temp[0],&Unu,&t1);
		gsl_blas_ddot(temp[1],var->Vnu,&t2);
		gsl_blas_ddot(var->Vnu,&Unu,&t3);

		lhood += -0.5/mod->cov*(uv*uv+t1+t2+t3)+rating/mod->cov*uv;
	}


    // 3.E[log p(z_n | \eta)] + E[log p(w_n | \beta)] + H(q(z_n | \phi_n))


    // 3.2 V
    lhood -= Vexpect_mult_norm(var) * Vdoc->total;
    for (i = 0; i < Vdoc->nterms; i++)
    {
        // !!! we can speed this up by turning it into a dot product
        // !!! profiler says this is where some time is spent
        for (j = 0; j < mod->k; j++)
        {
            double phi_ij = mget(var->Vphi, i, j);
            double log_phi_ij = mget(var->Vlog_phi, i, j);
            if (phi_ij > 0)
            {
                vinc(var->Vtopic_scores, j, phi_ij * Vdoc->count[i]);
                lhood +=
                    Vdoc->count[i] * phi_ij *
                    (vget(var->Vlambda, j) +
                     mget(mod->log_beta, j, Vdoc->word[i]) -
                     log_phi_ij);
            }
        }
    }

    var->lhood = lhood;
    assert(!isnan(var->lhood));
}


/**
 * optimize zeta
 *
 */

int opt_Uzeta(llna_var_param * var, llna_model * mod)
{
    int i;

    var->Uzeta = 0.0;
    //for (i = 0; i < mod->k-1; i++) //这里为什么是k-1，也是因为lambda(k)=0的缘故
    for (i = 0; i < mod->k; i++)
    	var->Uzeta += exp(vget(var->Ulambda, i) + (0.5) *  vget(var->Unu, i));
    return(0);
}

int opt_Vzeta(llna_var_param * var, llna_model * mod)
{
    int i;
    var->Vzeta = 0.0;
    for (i = 0; i < mod->k; i++)
    	var->Vzeta += exp(vget(var->Vlambda, i) + (0.5) *  vget(var->Vnu, i));
    return(0);
}


/**
 * optimize phi
 *
 */

void opt_Uphi(llna_var_param * var, doc * Udoc, llna_model * mod)
{
    int i, n, K = mod->k;
    double log_sum_n = 0;

    // compute phi proportions in log space

    for (n = 0; n < Udoc->nterms; n++)
    {
        log_sum_n = 0;
        for (i = 0; i < K; i++)
        {
            mset(var->Ulog_phi, n, i,
                 vget(var->Ulambda, i) + mget(mod->log_beta, i, Udoc->word[n]));
            if (i == 0)   //这里是否有问题？
                log_sum_n = mget(var->Ulog_phi, n, i);
            else
                log_sum_n =  log_sum(log_sum_n, mget(var->Ulog_phi, n, i));
        }
        for (i = 0; i < K; i++)
        {
            mset(var->Ulog_phi, n, i, mget(var->Ulog_phi, n, i) - log_sum_n);  //???
            mset(var->Uphi, n, i, exp(mget(var->Ulog_phi, n, i)));
        }
    }
}

void opt_Vphi(llna_var_param * var, doc * Vdoc, llna_model * mod)
{
    int i, n, K = mod->k;
    double log_sum_n = 0;

    // compute phi proportions in log space

    for (n = 0; n < Vdoc->nterms; n++)
    {
        log_sum_n = 0;
        for (i = 0; i < K; i++)
        {
            mset(var->Vlog_phi, n, i,
                 vget(var->Vlambda, i) + mget(mod->log_beta, i, Vdoc->word[n]));
            if (i == 0)
                log_sum_n = mget(var->Vlog_phi, n, i);
            else
                log_sum_n =  log_sum(log_sum_n, mget(var->Vlog_phi, n, i));
        }
        for (i = 0; i < K; i++)
        {
            mset(var->Vlog_phi, n, i, mget(var->Vlog_phi, n, i) - log_sum_n);  //???
            mset(var->Vphi, n, i, exp(mget(var->Vlog_phi, n, i)));
        }
    }
}

/**
 * optimize lambda
 *
 */

void fdf_Ulambda(const gsl_vector * p, void * params, double * f, gsl_vector * df)
{
    *f = f_Ulambda(p, params);
    df_Ulambda(p, params, df);
}
void fdf_Vlambda(const gsl_vector * p, void * params, double * f, gsl_vector * df)
{
    *f = f_Vlambda(p, params);
    df_Vlambda(p, params, df);
}

/*
 * 自变量p, 因为采用梯度下降法，因此采用 f = -L
 */


double f_Ulambda(const gsl_vector * p, void * params)  //目标函数中所有带 lambda的项, p 是变量,即lambda
{

    double term1, term2, term3, term4;
    int i,j;

    llna_var_param * var = ((bundle *) params)->var;
    doc * doc = ((bundle *) params)->Udoc;

    ratings * r_ui = ((bundle *) params)->r;
    llna_model * mod = ((bundle *) params)->mod;
    llna_corpus_var * c_var =  ((bundle *) params)->c_var;

    // compute lambda^T \sum phi
    gsl_blas_ddot(p,((bundle *) params)->Usum_phi, &term1);  //term1=p*sum_phi(向量点乘)

    // compute lambda - mu (= temp1)
    gsl_blas_dcopy(p, temp[1]);    //复制向量p 到 temp[1]
    gsl_blas_daxpy (-1.0, mod->Umu, temp[1]);  //temp=p-mu

    // compute (lambda - mu)^T Sigma^-1 (lambda - mu)
    gsl_blas_dsymv(CblasUpper, 1, mod->Uinv_cov, temp[1], 0, temp[2]);
    // gsl_blas_dgemv(CblasNoTrans, 1, mod->inv_cov, temp[1], 0, temp[2]);
    gsl_blas_ddot(temp[2], temp[1], &term2);
    term2 = - 0.5 * term2;

    // term3
    term3 = 0;
    for (i = 0; i < mod->k; i++)
        term3 += exp(vget(p, i) + (0.5) *  vget(var->Unu,i));
    term3 = -((1.0/var->Uzeta) * term3 - 1.0 + log(var->Uzeta)) * doc->total;

    // term4 -------只有这一项是我加的
    int item; double rating;
    gsl_vector Vlambda, Vnu;
    term4 = 0;
    double tt,t1,t2;
    for (i = 0; i<r_ui->nratings; i++ )
    {
    	item = r_ui->items[i];
    	rating = r_ui->ratings[i];
        Vlambda = gsl_matrix_row(c_var->Vcorpus_lambda, item-1).vector;
        Vnu = gsl_matrix_row(c_var->Vcorpus_nu, item-1).vector;

        gsl_blas_dcopy(p, temp[3]);    // temp[3] = lambda^2
        for (j = 0; j < mod->k; j++) {
        	tt = gsl_vector_get(temp[3],j)*gsl_vector_get(temp[3],j);
        	gsl_vector_set(temp[3],j,tt);
        }
        gsl_blas_ddot(p, &Vlambda, &t1);  //t1=Ulambda * Vlambda
        gsl_blas_ddot(temp[3],&Vnu, &t2);  //t2=Ulambda^2 *nu

        term4 += 0.5/mod->cov*(t1 * t1+t2)-rating/mod->cov*t1;

    }

    // negate for minimization
# if defined(DEBUG)
    printf("f_Ulambda=%f\n",-(term1+term2+term3-term4));
# endif

    return(-(term1+term2+term3-term4));
    //return(-(term1+term2+term3));
}
double f_Vlambda(const gsl_vector * p, void * params)  //目标函数中所有带 lambda的项, p 是变量,即lambda
{
    double term1, term2, term3, term4;
    int i,j;
    llna_var_param * var = ((bundle *) params)->var;
    doc * doc = ((bundle *) params)->Vdoc;
    ratings * r_vj = ((bundle *) params)->r;
    llna_model * mod = ((bundle *) params)->mod;
    llna_corpus_var * c_var =  ((bundle *) params)->c_var;


/*    printf("Vlambda=");
    for(i=0;i<mod->k;i++)
    	printf("%f\t",vget(p,i));
    printf("\n");*/

    // compute lambda^T \sum phi
    gsl_blas_ddot(p,((bundle *) params)->Vsum_phi, &term1);  //term1=p*sum_phi(向量点乘)

    // compute lambda - mu (= temp1)
    gsl_blas_dcopy(p, temp[1]);    //复制向量p 到 temp[1]
    gsl_blas_daxpy (-1.0, mod->Vmu, temp[1]);  //temp=p-mu

    // compute (lambda - mu)^T Sigma^-1 (lambda - mu)
    gsl_blas_dsymv(CblasUpper, 1, mod->Vinv_cov, temp[1], 0, temp[2]);
    // gsl_blas_dgemv(CblasNoTrans, 1, mod->inv_cov, temp[1], 0, temp[2]);
    gsl_blas_ddot(temp[2], temp[1], &term2);
    term2 = - 0.5 * term2;

    // term3
    term3 = 0;
    for (i = 0; i < mod->k; i++)
        term3 += exp(vget(p, i) + (0.5) *  vget(var->Vnu,i));
    term3 = -((1.0/var->Vzeta) * term3 - 1.0 + log(var->Vzeta)) * doc->total;

    // term4
    int user; double rating;
	gsl_vector Ulambda, Unu;
	term4 = 0;
	double tt,t1,t2;
	for (i = 0; i<r_vj->nratings; i++ )
	{
		user = r_vj->users[i];
		rating = r_vj->ratings[i];
		Ulambda = gsl_matrix_row(c_var->Ucorpus_lambda, user-1).vector;
		Unu = gsl_matrix_row(c_var->Ucorpus_nu, user-1).vector;

		gsl_blas_dcopy(p, temp[3]);    // temp[3] = lambda^2
		for (j = 0; j < mod->k; j++) {
			tt = gsl_vector_get(temp[3],j)*gsl_vector_get(temp[3],j);
			gsl_vector_set(temp[3],j,tt);
		}
		gsl_blas_ddot(p, &Ulambda, &t1);  //t1=Ulambda * Vlambda
		gsl_blas_ddot(temp[3],&Unu, &t2);  //t2=Ulambda^2 *nu

		term4 += 0.5/mod->cov*(t1 * t1+t2)-rating/mod->cov*t1;

	}

    // negate for minimization
# if defined(DEBUG)
	printf("f_Vlambda=%f\n",-(term1+term2+term3-term4));
# endif

    return(-(term1+term2+term3-term4));
}
/*
 * 自变量p, 因为采用梯度下降法，因此采用负导数 -dL/dlambda
 * this function should store the n-dimensional gradient df_i = d f(p,params) / d x_i
 */

void df_Ulambda(const gsl_vector * p, void * params, gsl_vector * df)
{
    // cast bundle {variational parameters, model, document}
	int i,j;

    llna_var_param * var = ((bundle *) params)->var;
    doc * doc = ((bundle *) params)->Udoc;
    ratings * r_ui = ((bundle *) params)->r;
    llna_corpus_var * c_var =  ((bundle *) params)->c_var;

    llna_model * mod = ((bundle *) params)->mod;
    gsl_vector * sum_phi = ((bundle *) params)->Usum_phi;


    //1. compute \Sigma^{-1} (\mu - \lambda) = temp[0]

    gsl_vector_set_zero(temp[0]);
    gsl_blas_dcopy(mod->Umu, temp[1]);
    gsl_vector_sub(temp[1], p);  //temp[1]=mu-lambda
    gsl_blas_dsymv(CblasLower, 1, mod->Uinv_cov, temp[1], 0, temp[0]); //temp[0]=inv*temp[1]

    //2. compute temp[1] 第二项 只有这一项是我加的

    double t1,tt;
    int item; double rating;
    gsl_vector Vlambda, Vnu;
    gsl_vector_set_all(temp[1],0.0);
    for (i = 0; i<r_ui->nratings; i++ )
    {
    	item = r_ui->items[i];
    	rating = r_ui->ratings[i];
        Vlambda = gsl_matrix_row(c_var->Vcorpus_lambda, item-1).vector;
        Vnu = gsl_matrix_row(c_var->Vcorpus_nu, item-1).vector;



        gsl_blas_dcopy(&Vlambda, temp[4]);
		gsl_blas_dcopy(p, temp[2]);

		gsl_blas_ddot(temp[4],p,&t1);  // t1=Ulambda*Vlambda
		gsl_vector_scale(temp[4], t1);  //temp[1]== Ulambda*Vlambda*Vlambda

		for (j = 0; j < mod->k; j++) {
			tt = gsl_vector_get(temp[2],j)*gsl_vector_get(&Vnu,j);
			gsl_vector_set(temp[2],j,tt); //temp[2] =lambda * nu
		}


		gsl_blas_dcopy(&Vlambda, temp[3]);
		gsl_vector_scale(temp[3],rating);  //temp[3]=y*Vlambda
		gsl_vector_add(temp[4],temp[2]);
		gsl_vector_sub(temp[4],temp[3]);
		gsl_vector_scale(temp[4],1.0/mod->cov);

		gsl_vector_add(temp[1],temp[4]);


    }


    //3. compute - (N / \zeta) * exp(\lambda + \nu^2 / 2)


    for (i = 0; i < temp[3]->size; i++)
    {
        vset(temp[3], i, -(((double) doc->total) / var->Uzeta) *
             exp(vget(p, i) + 0.5 * vget(var->Unu, i)));
    }

    // set return value (note negating derivative of bound)

    gsl_vector_set_all(df, 0.0);
    gsl_vector_sub(df, temp[0]); //df = df-temp[0]
    gsl_vector_sub(df, sum_phi);
    //gsl_vector_sub(df, temp[1]);
    gsl_vector_add(df, temp[1]);
    gsl_vector_sub(df, temp[3]);

# if defined(DEBUG)
    printf("df_Ulambda=");
    for (i=0;i<mod->k;i++) {
    	printf("%lf\t",vget(df,i));
    }
    printf("\n");
# endif
}

void df_Vlambda(const gsl_vector * p, void * params, gsl_vector * df)
{
    // cast bundle {variational parameters, model, document}
	int i,j;

    llna_var_param * var = ((bundle *) params)->var;
    doc * doc = ((bundle *) params)->Vdoc;
    ratings * r_vj = ((bundle *) params)->r;
    llna_corpus_var * c_var =  ((bundle *) params)->c_var;

    llna_model * mod = ((bundle *) params)->mod;
    gsl_vector * sum_phi = ((bundle *) params)->Vsum_phi;


    //1. compute \Sigma^{-1} (\mu - \lambda) = temp[0]

    gsl_vector_set_zero(temp[0]);
    gsl_blas_dcopy(mod->Vmu, temp[1]);
    gsl_vector_sub(temp[1], p);  //temp[1]=mu-lambda
    gsl_blas_dsymv(CblasLower, 1, mod->Vinv_cov, temp[1], 0, temp[0]); //temp[0]=inv*temp[1]


    //2. compute temp[1] 第二项 只有这一项是我加的
    double t1,tt;
	int user; double rating;
	gsl_vector Ulambda, Unu;
	gsl_vector_set_all(temp[1],0.0);
	for (i = 0; i<r_vj->nratings; i++ )
	{
		user = r_vj->users[i];
		rating = r_vj->ratings[i];
		Ulambda = gsl_matrix_row(c_var->Ucorpus_lambda, user-1).vector;
		Unu = gsl_matrix_row(c_var->Ucorpus_nu, user-1).vector;



		gsl_blas_dcopy(&Ulambda, temp[4]);
		gsl_blas_dcopy(p, temp[2]);

		gsl_blas_ddot(temp[4],p,&t1);  // t1=Ulambda*Vlambda
		gsl_vector_scale(temp[4], t1);  //temp[1]== Ulambda*Vlambda*Vlambda

		for (j = 0; j < mod->k; j++) {
			tt = gsl_vector_get(temp[2],j)*gsl_vector_get(&Unu,j);
			gsl_vector_set(temp[2],j,tt); //temp[2] =lambda * nu
		}


		gsl_blas_dcopy(&Ulambda, temp[3]);
		gsl_vector_scale(temp[3],rating);  //temp[3]=y*Vlambda
		gsl_vector_add(temp[4],temp[2]);
		gsl_vector_sub(temp[4],temp[3]);
		gsl_vector_scale(temp[4],1.0/mod->cov);

		gsl_vector_add(temp[1],temp[4]);


	}


    //3. compute - (N / \zeta) * exp(\lambda + \nu^2 / 2)


    for (i = 0; i < temp[3]->size; i++)
    {
        vset(temp[3], i, -(((double) doc->total) / var->Vzeta) *
             exp(vget(p, i) + 0.5 * vget(var->Vnu, i)));
    }

    // set return value (note negating derivative of bound)

    gsl_vector_set_all(df, 0.0);
    gsl_vector_sub(df, temp[0]); //df = df-temp[0]
    gsl_vector_sub(df, sum_phi);
    //gsl_vector_sub(df, temp[1]);
    gsl_vector_add(df, temp[1]);
    gsl_vector_sub(df, temp[3]);

# if defined(DEBUG)
    printf("df_Vlambda=");
    for (i=0;i<mod->k;i++)
    	printf("%lf\t",vget(df,i));
    printf("\n");
# endif
}

int opt_Ulambda(llna_var_param * var, doc * Udoc, ratings* r_ui,llna_model * mod, llna_corpus_var * c_var)
{
    gsl_multimin_function_fdf lambda_obj;
    const gsl_multimin_fdfminimizer_type * T;
    gsl_multimin_fdfminimizer * s;
    bundle b;
    int iter = 0, i, j;
    int status;
    double f_old, converged;

    //printf("%d,%d\n",r_ui->users[0],r_ui->nratings);
    b.var = var;
    b.Udoc = Udoc;
    //b.Vdoc = Vdoc;
    b.r = r_ui;
    b.c_var = c_var;
    b.mod = mod;

    // precompute \sum_n \phi_n and put it in the bundle

    b.Usum_phi = gsl_vector_alloc(mod->k); //为什么是k-1维？----------------------
    gsl_vector_set_zero(b.Usum_phi);
    for (i = 0; i < Udoc->nterms; i++) // nterms指的不同单词数
    {
        for (j = 0; j < mod->k; j++)
        {
            vset(b.Usum_phi, j,
                 vget(b.Usum_phi, j) +
                 ((double) Udoc->count[i]) * mget(var->Uphi, i, j)); //计算sum_phi-----参考公式（13）
        }
    }

    // part 1: Ulambda
    //共轭梯度法中主要的函数设置

    lambda_obj.f = &f_Ulambda;
    lambda_obj.df = &df_Ulambda;
    lambda_obj.fdf = &fdf_Ulambda;
    lambda_obj.n = mod->k;  //原论文中，由于最后一个lambda为0,因此不需要优化，
    lambda_obj.params = (void *)&b;

    // starting value
    // T = gsl_multimin_fdfminimizer_vector_bfgs;
    T = gsl_multimin_fdfminimizer_conjugate_fr;
    // T = gsl_multimin_fdfminimizer_steepest_descent;
    s = gsl_multimin_fdfminimizer_alloc (T, mod->k);

    gsl_vector* x = gsl_vector_calloc(mod->k);
    for (i = 0; i < mod->k; i++) vset(x, i, vget(var->Ulambda, i));


    gsl_multimin_fdfminimizer_set (s, &lambda_obj, x, 0.01, 1e-3);
    do
    {

# if defined(DEBUG)
    	// this part is for see the lhood
    	double fff=f_Ulambda(s->x, &b);
    	printf("confirm: f_Ulambda=%f\n",fff);
        for (i = 0; i < mod->k; i++)
        	vset(var->Ulambda, i, vget(s->x, i));  //------更新lambda-----
        Ulhood_bnd(var, Udoc, r_ui, mod, c_var);
        printf("\t\tlhood=%f\n",var->lhood);
        //
# endif


        iter++;
        f_old = s->f;
        status = gsl_multimin_fdfminimizer_iterate (s);
        converged = fabs((f_old - s->f) / f_old);
        //printf("f(lambda) = %5.17e ; conv = %5.17e\n", s->f, converged);
        if (status) break;
        status = gsl_multimin_test_gradient (s->gradient, PARAMS.cg_convergence); //判断是否收敛,(收敛时梯度应该为0)
        //printf("%f\n", gsl_blas_dnrm2(s->gradient));
    }
    while ((status == GSL_CONTINUE) &&
           ((PARAMS.cg_max_iter < 0) || (iter < PARAMS.cg_max_iter)));
    // while ((converged > PARAMS.cg_convergence) &&
    // ((PARAMS.cg_max_iter < 0) || (iter < PARAMS.cg_max_iter)));
    if (iter == PARAMS.cg_max_iter)
        printf("warning: cg didn't converge (lambda) \n");
    for (i = 0; i < mod->k; i++)
    	vset(var->Ulambda, i, vget(s->x, i));  //------更新lambda-----




/*    int item; double rating,yy;
	gsl_vector Vlambda, Vnu;
	for (i = 0; i<r_ui->nratings; i++ )
	{
		item = r_ui->items[i];
		rating = r_ui->ratings[i];
		Vlambda = gsl_matrix_row(c_var->Vcorpus_lambda, item-1).vector;
		gsl_blas_ddot(var->Ulambda,&Vlambda,&yy);
		printf("prediction= :%lf;\t true value=%lf\n",yy,rating);

	}*/


    gsl_multimin_fdfminimizer_free(s);
    gsl_vector_free(b.Usum_phi);
    gsl_vector_free(x);

    return(0);
}

int opt_Vlambda(llna_var_param * var, doc * Vdoc, ratings * r_vj, llna_model * mod, llna_corpus_var * c_var)
{
    gsl_multimin_function_fdf lambda_obj;
    const gsl_multimin_fdfminimizer_type * T;
    gsl_multimin_fdfminimizer * s;
    bundle b;
    int iter = 0, i, j;
    int status;
    double f_old, converged;

    b.var = var;
    b.Vdoc = Vdoc;
    b.r = r_vj;
    b.c_var = c_var;
    b.mod = mod;

    // precompute \sum_n \phi_n and put it in the bundle

    b.Vsum_phi = gsl_vector_alloc(mod->k); //为什么是k-1维？----------------------

    gsl_vector_set_zero(b.Vsum_phi);

    for (i = 0; i < Vdoc->nterms; i++) // nterms指的不同单词数
    {
        for (j = 0; j < mod->k; j++)
        {
            vset(b.Vsum_phi, j,
                 vget(b.Vsum_phi, j) +
                 ((double) Vdoc->count[i]) * mget(var->Vphi, i, j));
        }
    }
    // part 1: Ulambda
    //共轭梯度法中主要的函数设置

    lambda_obj.f = &f_Vlambda;
    lambda_obj.df = &df_Vlambda;
    lambda_obj.fdf = &fdf_Vlambda;
    lambda_obj.n = mod->k;  //原论文中，由于最后一个lambda为0,因此不需要优化，
    lambda_obj.params = (void *)&b;

    // starting value
    // T = gsl_multimin_fdfminimizer_vector_bfgs;
    T = gsl_multimin_fdfminimizer_conjugate_fr;
    // T = gsl_multimin_fdfminimizer_steepest_descent;
    s = gsl_multimin_fdfminimizer_alloc (T, mod->k);

    gsl_vector* x = gsl_vector_calloc(mod->k);
    for (i = 0; i < mod->k; i++) vset(x, i, vget(var->Vlambda, i));
    gsl_multimin_fdfminimizer_set (s, &lambda_obj, x, 0.01, 1e-3);
    do
    {
# if defined(DEBUG)
    	// this part is for see the lhood
    	double fff=f_Vlambda(s->x, &b);
    	printf("confirm: f_Vlambda=%f\n",fff);
        for (i = 0; i < mod->k; i++)
        	vset(var->Vlambda, i, vget(s->x, i));  //------更新lambda-----
        Vlhood_bnd(var, Vdoc, r_vj, mod, c_var);
        printf("\t\tlhood=%f\n",var->lhood);

    	//
# endif

        iter++;
        f_old = s->f;
        status = gsl_multimin_fdfminimizer_iterate (s);
        converged = fabs((f_old - s->f) / f_old);
        //printf("f(lambda) = %5.17e ; conv = %5.17e\n", s->f, converged);
        if (status) break;
        status = gsl_multimin_test_gradient (s->gradient, PARAMS.cg_convergence); //判断是否收敛,(收敛时梯度应该为0)
        //printf("%f\n", gsl_blas_dnrm2(s->gradient));
    }
    while ((status == GSL_CONTINUE) &&
           ((PARAMS.cg_max_iter < 0) || (iter < PARAMS.cg_max_iter)));
    // while ((converged > PARAMS.cg_convergence) &&
    // ((PARAMS.cg_max_iter < 0) || (iter < PARAMS.cg_max_iter)));
    if (iter == PARAMS.cg_max_iter)
        printf("warning: cg didn't converge (lambda) \n");


    for (i = 0; i < mod->k; i++)
    	vset(var->Vlambda, i, vget(s->x, i));  //------更新lambda-----
/*

    printf("Vlambda = ");
    for (i = 0; i < mod->k; i++)
    	printf("%lf\t",vget(var->Vlambda, i));   //竟然全是0？
    printf("\n");
*/



    //vset(var->Vlambda, i, 0);

    gsl_multimin_fdfminimizer_free(s);
    gsl_vector_free(b.Vsum_phi);
    gsl_vector_free(x);

    return(0);
}

/**
 * optimize nu
 *
 */

double f_nu_i(double nu_i, int i, llna_var_param * var,
              llna_model * mod, doc * d)
{
/*    double v;

    v = - (nu_i * mget(mod->inv_cov, i, i) * 0.5)
        - (((double) d->total/var->zeta) * exp(vget(var->lambda, i) + nu_i/2))
        + (0.5 * safe_log(nu_i));*/

    return(0);
}

//自变量nu_i
double df_Unu_i(double nu_i, int i, llna_var_param * var, ratings * r_ui,
               llna_model * mod, doc * d,llna_corpus_var * c_var)
{
    double v, myterm=0;

    int j,item; double rating;
    gsl_vector Vlambda, Vnu;

    double tt,t1,t2;
    for (j = 0; j<r_ui->nratings; j++ )
    {
    	item = r_ui->items[j];
		Vlambda = gsl_matrix_row(c_var->Vcorpus_lambda, item-1).vector;
		Vnu = gsl_matrix_row(c_var->Vcorpus_nu, item-1).vector;

    	myterm += 0.5/mod->cov*(vget(&Vlambda, i)*vget(&Vlambda, i)+vget(&Vnu, i));

    }

    v = - (mget(mod->Uinv_cov, i, i) * 0.5)
        - (0.5 * ((double) d->total/var->Uzeta) * exp(vget(var->Ulambda, i) + nu_i/2))
        + (0.5 * (1.0 / nu_i))
        - myterm;

    return(v);
}

double df_Vnu_i(double nu_i, int i, llna_var_param * var, ratings * r_vj,
               llna_model * mod, doc * d, llna_corpus_var * c_var)
{
    double v, myterm=0;
    int j, user; double rating;
    gsl_vector Ulambda, Unu;

    double tt,t1,t2;
    for (j = 0; j<r_vj->nratings; j++ )
    {
    	user = r_vj->users[j];
		Ulambda = gsl_matrix_row(c_var->Ucorpus_lambda, user-1).vector;
		Unu = gsl_matrix_row(c_var->Ucorpus_nu, user-1).vector;

    	myterm += 0.5/mod->cov*(vget(&Ulambda, i)*vget(&Ulambda, i)+vget(&Unu, i));

    }


    v = - (mget(mod->Vinv_cov, i, i) * 0.5)
        - (0.5 * ((double) d->total/var->Vzeta) * exp(vget(var->Vlambda, i) + nu_i/2))
        + (0.5 * (1.0 / nu_i))
        - myterm;

    return(v);
}

double d2f_Unu_i(double nu_i, int i, llna_var_param * var, llna_model * mod, doc * d)
{
    double v;

    v = - (0.25 * ((double) d->total/var->Uzeta) * exp(vget(var->Ulambda, i) + nu_i/2))
        - (0.5 * (1.0 / (nu_i * nu_i)));

    return(v);
}

double d2f_Vnu_i(double nu_i, int i, llna_var_param * var, llna_model * mod, doc * d)
{
    double v;

    v = - (0.25 * ((double) d->total/var->Vzeta) * exp(vget(var->Vlambda, i) + nu_i/2))
        - (0.5 * (1.0 / (nu_i * nu_i)));

    return(v);
}

void opt_Unu(llna_var_param * var, doc * Udoc, ratings * r_ui,llna_model * mod, llna_corpus_var * c_var )
{
    int i;

    // !!! here i changed to k-1
    for (i = 0; i < mod->k; i++)
    	opt_Unu_i(i, var, r_ui, mod, Udoc,c_var);

}

void opt_Vnu(llna_var_param * var, doc * Vdoc, ratings * r_vj,llna_model * mod, llna_corpus_var * c_var)
{
    int i;

    // !!! here i changed to k-1
    for (i = 0; i < mod->k; i++)
    	opt_Vnu_i(i, var, r_vj, mod, Vdoc,c_var);

}



double fixed_point_iter_i(int i, llna_var_param * var, llna_model * mod, doc * d)
{
/*    double v;
    double lambda = vget(var->lambda, i);
    double nu = vget(var->nu, i);
    double c = ((double) d->total / var->zeta);

    v = mget(mod->inv_cov,i,i) + c * exp(lambda + nu/2);*/

    return(0);
}


void opt_Unu_i(int i, llna_var_param * var, ratings * r_ui, llna_model * mod, doc * d ,llna_corpus_var * c_var)
{
    double init_nu = 10;
    double nu_i = 0, log_nu_i = 0, df = 0, d2f = 0;
    int iter = 0;

    log_nu_i = log(init_nu);
    do
    {
        iter++;
        nu_i = exp(log_nu_i);
        // assert(!isnan(nu_i));
        if (isnan(nu_i))
        {
            init_nu = init_nu*2;
            printf("warning : nu is nan; new init = %5.5f\n", init_nu);
            log_nu_i = log(init_nu);
            nu_i = init_nu;
        }
        // f = f_nu_i(nu_i, i, var, mod, d);
        // printf("%5.5f  %5.5f \n", nu_i, f);
        df = df_Unu_i(nu_i, i, var, r_ui, mod, d, c_var);
        //printf("%f\n",df);
        d2f = d2f_Unu_i(nu_i, i, var, mod, d);
        log_nu_i = log_nu_i - (df*nu_i)/(d2f*nu_i*nu_i+df*nu_i);
    }
    while (fabs(df) > NEWTON_THRESH && iter < 100);

    vset(var->Unu, i, exp(log_nu_i));
}

void opt_Vnu_i(int i, llna_var_param * var, ratings * r_vj, llna_model * mod, doc * d,llna_corpus_var * c_var)
{
    double init_nu = 10;
    double nu_i = 0, log_nu_i = 0, df = 0, d2f = 0;
    int iter = 0;

    log_nu_i = log(init_nu);
    do
    {
        iter++;
        nu_i = exp(log_nu_i);
        // assert(!isnan(nu_i));
        if (isnan(nu_i))
        {
            init_nu = init_nu*2;
            printf("warning : nu is nan; new init = %5.5f\n", init_nu);
            log_nu_i = log(init_nu);
            nu_i = init_nu;
        }
        // f = f_nu_i(nu_i, i, var, mod, d);
        // printf("%5.5f  %5.5f \n", nu_i, f);
        df = df_Vnu_i(nu_i, i, var, r_vj,mod, d,c_var);
        d2f = d2f_Vnu_i(nu_i, i, var, mod, d);
        log_nu_i = log_nu_i - (df*nu_i)/(d2f*nu_i*nu_i+df*nu_i);  //这里是梯度上升法，不知道更新公式对不对？
    }
    while (fabs(df) > NEWTON_THRESH && iter < 100);

    vset(var->Vnu, i, exp(log_nu_i));
}

/**
 * initial variational parameters
 *
 */

void init_Uvar_unif(llna_var_param * var, doc * Udoc, llna_model * mod)
{
    int i;

    gsl_matrix_set_all(var->Uphi, 1.0/mod->k);
    gsl_matrix_set_all(var->Ulog_phi, -log((double) mod->k));

    var->Uzeta = 10;

    //for (i = 0; i < mod->k-1; i++)
    for (i = 0; i < mod->k; i++)
    {
        //vset(var->nu, i, 10.0);
        vset(var->Ulambda, i, 1.0);
        vset(var->Unu, i, 1.0);
    }
    var->niter = 0;
    var->lhood = 0;
}
void init_Vvar_unif(llna_var_param * var, doc * Vdoc, llna_model * mod)
{
    int i;
    gsl_matrix_set_all(var->Vphi, 1.0/mod->k);
    gsl_matrix_set_all(var->Vlog_phi, -log((double) mod->k));
    var->Vzeta = 10;
    for (i = 0; i < mod->k; i++)
    {
        vset(var->Vlambda, i, 1.0);
        vset(var->Vnu, i, 1.0);
    }
    var->niter = 0;
    var->lhood = 0;
}

ratings*  get_uratings(ratings * r, int nusers)
{
	ratings* r_u;
	int i,nr,user;


	r_u = malloc(sizeof(ratings)*nusers);
	for(user=1; user<=nusers; user++)
	{


		//r_u->users = malloc(sizeof(int)*1);
		//r_u[user-1].users = malloc(sizeof(int)*1);

		r_u[user-1].users = malloc(sizeof(int)*1);
		r_u[user-1].items = malloc(sizeof(int)*1);
		r_u[user-1].ratings = malloc(sizeof(double)*1);
		r_u[user-1].nratings = 0;
	}
	nr = 0;



	for (i = 0; i < r->nratings; i++)
	{
		//printf("%d,%d,%lf\n",r->users[i],r->items[i],r->ratings[i]);
		user = r->users[i];
		nr = r_u[user-1].nratings;

		r_u[user-1].users = (int*) realloc(r_u[user-1].users, sizeof(int)*(nr+1));
		r_u[user-1].items = (int*) realloc(r_u[user-1].items, sizeof(int)*(nr+1));
		r_u[user-1].ratings = (double*) realloc(r_u[user-1].ratings, sizeof(double)*(nr+1));


		r_u[user-1].users[nr] = r->users[i];
		r_u[user-1].items[nr] = r->items[i];
		r_u[user-1].ratings[nr] = r->ratings[i];

		r_u[user-1].nratings++;
		//printf("%d\n",r_u[user-1].users[nr]);
	}
	 //printf("%d\n",r_u[0].users[0]);
	 return(r_u);

}


ratings * get_vratings(ratings * r, int nitems)
{
	ratings* r_v;
	int i,nr,item;
	r_v = malloc(sizeof(ratings)*nitems);
	for(item=1; item<=nitems; item++)
	{
		r_v[item-1].users = malloc(sizeof(int)*1);
		r_v[item-1].items = malloc(sizeof(int)*1);
		r_v[item-1].ratings = malloc(sizeof(double)*1);
		r_v[item-1].nratings = 0;
	}
	nr = 0;

	for (i = 0; i < r->nratings; i++)
	{
		//printf("%d,%d,%lf\n",r->users[i],r->items[i],r->ratings[i]);
		item = r->items[i];
		nr = r_v[item-1].nratings;

		r_v[item-1].users = (int*) realloc(r_v[item-1].users, sizeof(int)*(nr+1));
		r_v[item-1].items = (int*) realloc(r_v[item-1].items, sizeof(int)*(nr+1));
		r_v[item-1].ratings = (double*) realloc(r_v[item-1].ratings, sizeof(double)*(nr+1));


		r_v[item-1].users[nr] = r->users[i];
		r_v[item-1].items[nr] = r->items[i];
		r_v[item-1].ratings[nr] = r->ratings[i];

		r_v[item-1].nratings++;

		//printf("%d\n",r_v[item-1].items[nr]);
	}
	return(r_v);
}

void init_Uvar(llna_var_param * var, doc * Udoc, llna_model * mod, gsl_vector *Ulambda, gsl_vector* Unu)
{
    gsl_vector_memcpy(var->Ulambda, Ulambda);
    gsl_vector_memcpy(var->Unu, Unu);

    opt_Uzeta(var, mod);
    opt_Uphi(var, Udoc, mod);
    var->niter = 0;
}
void init_Vvar(llna_var_param * var, doc * Vdoc, llna_model * mod, gsl_vector *Vlambda, gsl_vector* Vnu)
{
    gsl_vector_memcpy(var->Vlambda, Vlambda);
    gsl_vector_memcpy(var->Vnu, Vnu);

    opt_Vzeta(var,  mod);
    opt_Vphi(var, Vdoc, mod);
    var->niter = 0;
}




/**
 *
 * variational inference
 *
 */

llna_var_param * new_llna_Uvar_param(int Unterms, int k)
{
    llna_var_param * ret = malloc(sizeof(llna_var_param));
    ret->Ulambda = gsl_vector_alloc(k);
    ret->Unu = gsl_vector_alloc(k);
    //ret->Unu = malloc(sizeof(double)); //因为不是指针，因此就没必要申请空间
    ret->Uphi = gsl_matrix_alloc(Unterms, k);
    ret->Ulog_phi = gsl_matrix_alloc(Unterms, k);

    ret->Uzeta = 0;

    ret->Utopic_scores = gsl_vector_alloc(k); //???干嘛的
    return(ret);
}


llna_var_param * new_llna_Vvar_param(int Vnterms, int k)
{
    llna_var_param * ret = malloc(sizeof(llna_var_param));
    ret->Vlambda = gsl_vector_alloc(k);
    ret->Vnu = gsl_vector_alloc(k);
    ret->Vphi = gsl_matrix_alloc(Vnterms, k);
    ret->Vlog_phi = gsl_matrix_alloc(Vnterms, k);
    ret->Vzeta = 0;
    ret->Vtopic_scores = gsl_vector_alloc(k); //???干嘛的
    return(ret);
}
llna_corpus_var * new_llna_corpus_var(int nusers, int nitems, int k)
{
	llna_corpus_var * c_var = malloc(sizeof(llna_corpus_var));
    c_var->Ucorpus_lambda = gsl_matrix_alloc(nusers, k);
    c_var->Vcorpus_lambda = gsl_matrix_alloc(nitems, k);

    c_var->Ucorpus_nu = gsl_matrix_alloc(nusers, k);
    c_var->Vcorpus_nu = gsl_matrix_alloc(nitems, k);


    c_var->Ucorpus_phi_sum = gsl_matrix_alloc(nusers, k);
    c_var->Vcorpus_phi_sum = gsl_matrix_alloc(nitems, k);
    return(c_var);
}
void  init_corpus_var(llna_corpus_var * c_var, char* start)
{
	if (strcmp(start, "rand")==0) {
		gsl_matrix_set_all(c_var->Ucorpus_lambda,1.0);
		gsl_matrix_set_all(c_var->Vcorpus_lambda,1.0);

		gsl_matrix_set_all(c_var->Ucorpus_nu,1.0);
		gsl_matrix_set_all(c_var->Vcorpus_nu,1.0);
	} else {
		char fname[100];
		sprintf(fname, "%s-Ucorpus_lambda.dat", start);
		scanf_matrix(fname, c_var->Ucorpus_lambda);
        sprintf(fname, "%s-Vcorpus_lambda.dat", start);
        scanf_matrix(fname, c_var->Vcorpus_lambda);

		sprintf(fname, "%s-Ucorpus_nu.dat", start);
		scanf_matrix(fname, c_var->Ucorpus_nu);
        sprintf(fname, "%s-Vcorpus_nu.dat", start);
        scanf_matrix(fname, c_var->Vcorpus_nu);
	}


}

void free_llna_Uvar_param(llna_var_param * v)
{
    gsl_vector_free(v->Ulambda);
    gsl_vector_free(v->Unu);
    gsl_matrix_free(v->Uphi);
    gsl_matrix_free(v->Ulog_phi);
    gsl_vector_free(v->Utopic_scores);
    free(v);
}
void free_llna_Vvar_param(llna_var_param * v)
{
    gsl_vector_free(v->Vlambda);
    gsl_vector_free(v->Vnu);
    gsl_matrix_free(v->Vphi);
    gsl_matrix_free(v->Vlog_phi);
    gsl_vector_free(v->Vtopic_scores);
    free(v);
}

// 这里是针对一个user文档，一个item文档的
double Uvar_inference(llna_var_param * var,
                     doc * Udoc, ratings * r_ui,
                     llna_model * mod, llna_corpus_var * c_var)
{
    double lhood_old = 0;
    double convergence;

    Ulhood_bnd(var, Udoc, r_ui, mod, c_var);  //这里输出mod->lhood
    do
    {
        var->niter++;
        opt_Uzeta(var, mod);
        //printf("function opt_zeta done\n");
        opt_Ulambda(var, Udoc, r_ui, mod, c_var);  //比较复杂
        opt_Uzeta(var, mod);
        //printf("function opt_zeta2 done\n");
        opt_Unu(var, Udoc,r_ui, mod, c_var);      //比较复杂
        //printf("function opt_nu done\n");
        opt_Uzeta(var, mod);
        //printf("function opt_zeta3 done\n");
        opt_Uphi(var, Udoc, mod); // 这里面的细节貌似有问题
        //opt_Vphi(var, Udoc, mod);
        //printf("function opt_phi done\n");

        lhood_old = var->lhood;
        //lhood_bnd(var, Udoc, Vdoc,rating, mod);  //重新计算 mod->lhood，也就是下界。当下界稳定时，就停止迭代
        Ulhood_bnd(var, Udoc, r_ui, mod, c_var);
        convergence = fabs((lhood_old - var->lhood) / lhood_old);
        // printf("lhood = %8.6f (%7.6f)\n", var->lhood, convergence);

        if ((lhood_old > var->lhood) && (var->niter > 1))
            printf("WARNING: iter %05d %5.5f > %5.5f\n",
                   var->niter, lhood_old, var->lhood);
    }
    while ((convergence > PARAMS.var_convergence) &&
           ((PARAMS.var_max_iter < 0) || (var->niter < PARAMS.var_max_iter)));

    if (convergence > PARAMS.var_convergence) var->converged = 0;
    else var->converged = 1;

#if defined(SHOW_PREDICTION)
    int i,item; double rating,yy;
    gsl_vector Vlambda, Vnu;
    for (i = 0; i<r_ui->nratings; i++ )
    {
    	item = r_ui->items[i];
		rating = r_ui->ratings[i];
		Vlambda = gsl_matrix_row(c_var->Vcorpus_lambda, item-1).vector;
		gsl_blas_ddot(var->Ulambda,&Vlambda,&yy);
		printf("prediction= :%lf;\t true value=%lf\n",yy,rating);

    }
#endif

    return(var->lhood);
}

double Vvar_inference(llna_var_param * var,
                     doc * Vdoc, ratings * r_vj,
                     llna_model * mod, llna_corpus_var * c_var)
{

    double lhood_old = 0;
    double convergence;

    Vlhood_bnd(var, Vdoc, r_vj, mod, c_var);  //这里输出mod->lhood
    do
    {
        var->niter++;
        opt_Vzeta(var, mod);
        opt_Vlambda(var, Vdoc, r_vj, mod, c_var);  //比较复杂
        opt_Vzeta(var, mod);
        opt_Vnu(var, Vdoc,r_vj, mod, c_var);      //比较复杂
        opt_Vzeta(var, mod);
        opt_Vphi(var, Vdoc, mod);
        lhood_old = var->lhood;
        //lhood_bnd(var, Udoc, Vdoc,rating, mod);  //重新计算 mod->lhood，也就是下界。当下界稳定时，就停止迭代
        Vlhood_bnd(var, Vdoc, r_vj, mod, c_var);
        convergence = fabs((lhood_old - var->lhood) / lhood_old);
        // printf("lhood = %8.6f (%7.6f)\n", var->lhood, convergence);

        if ((lhood_old > var->lhood) && (var->niter > 1))
            printf("WARNING: iter %05d %5.5f > %5.5f\n",
                   var->niter, lhood_old, var->lhood);
    }
    while ((convergence > PARAMS.var_convergence) &&
           ((PARAMS.var_max_iter < 0) || (var->niter < PARAMS.var_max_iter)));

    if (convergence > PARAMS.var_convergence) var->converged = 0;
    else var->converged = 1;

#if defined(SHOW_PREDICTION)
    int i,user; double rating,yy;
    gsl_vector Ulambda, Unu;
    for (i = 0; i<r_vj->nratings; i++ )
    {
    	user = r_vj->users[i];
		rating = r_vj->ratings[i];
		Ulambda = gsl_matrix_row(c_var->Ucorpus_lambda, user-1).vector;
		gsl_blas_ddot(var->Vlambda,&Ulambda,&yy);
		printf("prediction= :%lf;\t true value=%lf\n",yy,rating);

    }
#endif

    return(var->lhood);
}

void update_Uexpected_ss(llna_var_param* var, doc* Udoc, ratings * r_ui, llna_corpus_var * c_var,llna_ss* ss)
{
    int i, j, w, c;
    double lilj;

    // 1. covariance and mean suff stats
    for (i = 0; i < ss->Ucov_ss->size1; i++)
    {
        vinc(ss->Umu_ss, i, vget(var->Ulambda, i));  //gsl_vector_get
        for (j = 0; j < ss->Ucov_ss->size2; j++)
        {
            lilj = vget(var->Ulambda, i) * vget(var->Ulambda, j);
            if (i==j)
                mset(ss->Ucov_ss, i, j,
                     mget(ss->Ucov_ss, i, j) + vget(var->Unu, i) + lilj);
            else
                mset(ss->Ucov_ss, i, j, mget(ss->Ucov_ss, i, j) + lilj);
        }
    }
    // 2. topics suff stats
    for (i = 0; i < Udoc->nterms; i++)
    {
        for (j = 0; j < ss->Ubeta_ss->size1; j++)
        {
            w = Udoc->word[i];
            c = Udoc->count[i];
            mset(ss->Ubeta_ss, j, w,
                 mget(ss->Ubeta_ss, j, w) + c * mget(var->Uphi, i, j));
        }
    }

    // 3. ratings covariance suff stats
    int item;
    double rating,uv,t1,t2,t3;
    gsl_vector Vlambda, Vnu;

    gsl_blas_dcopy(var->Ulambda,temp[0]);  //temp[0]=Ulambda^2
    for (i = 0; i < var->Ulambda->size ; i++) {
    	t1 = gsl_vector_get(temp[0],i)*gsl_vector_get(temp[0],i);
    	gsl_vector_set(temp[0],i,t1);
    }
    for(i = 0; i < r_ui->nratings; i++)
    {
    	item = r_ui->items[i];
    	rating = r_ui->ratings[i];
        Vlambda = gsl_matrix_row(c_var->Vcorpus_lambda, item-1).vector;
        Vnu = gsl_matrix_row(c_var->Vcorpus_nu, item-1).vector;

        gsl_blas_ddot(var->Ulambda,&Vlambda,&uv);
        gsl_blas_dcopy(&Vlambda,temp[1]);  //temp[1]=Vlambda^2
        for (j = 0; j < Vlambda.size ; j++) {
        	t1 = gsl_vector_get(temp[1],j)*gsl_vector_get(temp[1],j);
        	gsl_vector_set(temp[1],j,t1);
        }
        gsl_blas_ddot(temp[0],&Vnu, &t1);  //t1=Ulambda^2 *Vnu
        gsl_blas_ddot(temp[1],var->Unu, &t2);  //t2=Vlambda^2 *Unu
        gsl_blas_ddot(var->Unu,&Vnu, &t3);  //t3=Unu * Vnu

        ss->cov_ss += rating*rating + uv*uv + t1 + t2 + t3 - 2*rating*uv;
        ss->nratings++;

    }




    // 4. number of data
    ss->Undata++;  //最终应该等于 nratings
}

void update_Vexpected_ss(llna_var_param* var, doc* Vdoc, ratings * r_vj,llna_corpus_var * c_var, llna_ss* ss)
{
    int i, j, w, c;
    double lilj;

    // 1.covariance and mean suff stats
    for (i = 0; i < ss->Vcov_ss->size1; i++)
    {
        vinc(ss->Vmu_ss, i, vget(var->Vlambda, i));  //gsl_vector_get
        for (j = 0; j < ss->Vcov_ss->size2; j++)
        {
            lilj = vget(var->Vlambda, i) * vget(var->Vlambda, j);
            if (i==j)
                mset(ss->Vcov_ss, i, j,
                     mget(ss->Vcov_ss, i, j) + vget(var->Vnu, i) + lilj);
            else
                mset(ss->Vcov_ss, i, j, mget(ss->Vcov_ss, i, j) + lilj);
        }
    }
    // 2.topics suff stats
    for (i = 0; i < Vdoc->nterms; i++)
    {
        for (j = 0; j < ss->Vbeta_ss->size1; j++)
        {
            w = Vdoc->word[i];
            c = Vdoc->count[i];
            mset(ss->Vbeta_ss, j, w,
                 mget(ss->Vbeta_ss, j, w) + c * mget(var->Vphi, i, j));
        }
    }

    // 3. ratings covariance suff stats
    int user;
    double rating,uv,t1,t2,t3;
    gsl_vector Ulambda, Unu;

    gsl_blas_dcopy(var->Vlambda,temp[0]);  //temp[0]=Vlambda^2
    for (i = 0; i < var->Vlambda->size ; i++) {
    	t1 = gsl_vector_get(temp[0],i)*gsl_vector_get(temp[0],i);
    	gsl_vector_set(temp[0],i,t1);
    }
    for(i = 0; i < r_vj->nratings; i++)
    {
    	user = r_vj->users[i];
    	rating = r_vj->ratings[i];
        Ulambda = gsl_matrix_row(c_var->Ucorpus_lambda, user-1).vector;
        Unu = gsl_matrix_row(c_var->Ucorpus_nu, user-1).vector;

        gsl_blas_ddot(&Ulambda,var->Vlambda,&uv);
        gsl_blas_dcopy(&Ulambda,temp[1]);  //temp[1]=Ulambda^2
        for (j = 0; j < Ulambda.size ; j++) {
        	t1 = gsl_vector_get(temp[1],j)*gsl_vector_get(temp[1],j);
        	gsl_vector_set(temp[1],j,t1);
        }
        gsl_blas_ddot(temp[0],&Unu, &t1);  //t1=Ulambda^2 *Vnu
        gsl_blas_ddot(temp[1],var->Vnu, &t2);  //t2=Vlambda^2 *Unu
        gsl_blas_ddot(&Unu,var->Vnu, &t3);  //t3=Unu * Vnu

        ss->cov_ss += rating*rating + uv*uv + t1 + t2 + t3 - 2*rating*uv;
        ss->nratings++;

    }

    // 4. number of data
    ss->Vndata++;  //最终应该等于 nratings
}

/*
 * importance sampling the likelihood based on the variational posterior
 *
 */

double sample_term(llna_var_param* var, doc* d, llna_model* mod, double* eta)
{/*
    int i, j, n;
    double t1, t2, sum, theta[mod->k];
    double word_term;

    t1 = (0.5) * mod->log_det_inv_cov;
    t1 += - (0.5) * (mod->k) * 1.837877;
    for (i = 0; i < mod->k; i++)
        for (j = 0; j < mod->k ; j++)
            t1 -= (0.5) *
                (eta[i] - vget(mod->mu, i)) *
                mget(mod->inv_cov, i, j) *
                (eta[j] - vget(mod->mu, j));

    // compute theta
    sum = 0;
    for (i = 0; i < mod->k; i++)
    {
        theta[i] = exp(eta[i]);
        sum += theta[i];
    }
    for (i = 0; i < mod->k; i++)
    {
        theta[i] = theta[i] / sum;
    }

    // compute word probabilities
    for (n = 0; n < d->nterms; n++)
    {
        word_term = 0;
        for (i = 0; i < mod->k; i++)
            word_term += theta[i]*exp(mget(mod->log_beta,i,d->word[n]));
        t1 += d->count[n] * safe_log(word_term);
    }

    // log(q(\eta | lambda, nu))
    t2 = 0;
    for (i = 0; i < mod->k; i++)
        t2 += log(gsl_ran_gaussian_pdf(eta[i] - vget(var->lambda,i), sqrt(vget(var->nu,i))));
    return(t1-t2);*/
	return(0);
}


double sample_lhood(llna_var_param* var, doc* d, llna_model* mod)
{/*
    int nsamples, i, n;
    double eta[mod->k];
    double log_prob, sum = 0, v;
    gsl_rng * r = gsl_rng_alloc(gsl_rng_taus);

    gsl_rng_set(r, (long) 1115574245);
    nsamples = 10000;

    // for each sample
    for (n = 0; n < nsamples; n++)
    {
        // sample eta from q(\eta)
        for (i = 0; i < mod->k; i++)
        {
            v = gsl_ran_gaussian_ratio_method(r, sqrt(vget(var->nu,i)));
            eta[i] = v + vget(var->lambda, i);
        }
        // compute p(w | \eta) - q(\eta)
        log_prob = sample_term(var, d, mod, eta);
        // update log sum
        if (n == 0) sum = log_prob;
        else sum = log_sum(sum, log_prob);
        // printf("%5.5f\n", (sum - log(n+1)));
    }
    sum = sum - log((double) nsamples);
    return(sum);*/
	return(0);
}


/*
 * expected theta under a variational distribution
 *
 * (v is assumed allocated to the right length.)
 *
 */


void expected_theta(llna_var_param *var, doc* d, llna_model *mod, gsl_vector* val)
{/*
    int nsamples, i, n;
    double eta[mod->k];
    double theta[mod->k];
    double e_theta[mod->k];
    double sum, w, v;
    gsl_rng * r = gsl_rng_alloc(gsl_rng_taus);

    gsl_rng_set(r, (long) 1115574245);
    nsamples = 100;

    // initialize e_theta
    for (i = 0; i < mod->k; i++) e_theta[i] = -1;
    // for each sample
    for (n = 0; n < nsamples; n++)
    {
        // sample eta from q(\eta)
        for (i = 0; i < mod->k; i++)
        {
            v = gsl_ran_gaussian_ratio_method(r, sqrt(vget(var->nu,i)));
            eta[i] = v + vget(var->lambda, i);
        }
        // compute p(w | \eta) - q(\eta)
        w = sample_term(var, d, mod, eta);
        // compute theta
        sum = 0;
        for (i = 0; i < mod->k; i++)
        {
            theta[i] = exp(eta[i]);
            sum += theta[i];
        }
        for (i = 0; i < mod->k; i++)
            theta[i] = theta[i] / sum;
        // update e_theta
        for (i = 0; i < mod->k; i++)
            e_theta[i] = log_sum(e_theta[i], w +  safe_log(theta[i]));
    }
    // normalize e_theta and set return vector
    sum = -1;
    for (i = 0; i < mod->k; i++)
    {
        e_theta[i] = e_theta[i] - log(nsamples);
        sum = log_sum(sum, e_theta[i]);
    }
    for (i = 0; i < mod->k; i++)
        vset(val, i, exp(e_theta[i] - sum));*/
}

/*
 * log probability of the document under proportions theta and topics
 * beta
 *
 */

double log_mult_prob(doc* d, gsl_vector* theta, gsl_matrix* log_beta)
{
    int i, k;
    double ret = 0;
    double term_prob;

    for (i = 0; i < d->nterms; i++)
    {
        term_prob = 0;
        for (k = 0; k < log_beta->size1; k++)
            term_prob += vget(theta, k) * exp(mget(log_beta, k, d->word[i]));
        ret = ret + safe_log(term_prob) * d->count[i];
    }
    return(ret);
}


/*
 * writes the word assignments line for a document to a file
 *
 */

void write_word_assignment(FILE* f, doc* d, gsl_matrix* phi)
{
    int n;

    fprintf(f, "%03d", d->nterms);
    for (n = 0; n < d->nterms; n++)
    {
        gsl_vector phi_row = gsl_matrix_row(phi, n).vector;
        fprintf(f, " %04d:%02d", d->word[n], argmax(&phi_row));
    }
    fprintf(f, "\n");
    fflush(f);
}

/*
 * write corpus variational parameter
 *
 */

void write_c_var(llna_corpus_var * c_var, char * root)  //已经改完
{
    char filename[200];

    sprintf(filename, "%s-Ucorpus_lambda.dat", root);
    printf_matrix(filename, c_var->Ucorpus_lambda);

    sprintf(filename, "%s-Vcorpus_lambda.dat", root);
    printf_matrix(filename, c_var->Vcorpus_lambda);

    sprintf(filename, "%s-Ucorpus_nu.dat", root);
    printf_matrix(filename, c_var->Ucorpus_nu);

    sprintf(filename, "%s-Vcorpus_nu.dat", root);
    printf_matrix(filename, c_var->Vcorpus_nu);

}
