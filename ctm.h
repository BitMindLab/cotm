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


#ifndef LLNA_H
#define LLNA_H

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <time.h>
#include "corpus.h"

#define NUM_INIT 1
#define SEED_INIT_SMOOTH 1.0

/*
 * the llna model
 *
 */

typedef struct llna_model
{
    int k;
    gsl_matrix * log_beta;

    //gsl_vector * mu;
    gsl_vector * Umu;
    gsl_vector * Vmu;

    //double inv_cov;
    gsl_matrix * Uinv_cov;
    gsl_matrix * Vinv_cov;

    double cov;
    gsl_matrix * Ucov;
    gsl_matrix * Vcov;

    //double log_det_inv_cov;
    double Ulog_det_inv_cov;
    double Vlog_det_inv_cov;
} llna_model;  //这是 模型参数（不包括隐变量）


/*
 * sufficient statistics for mle of an llna model
 *
 */

typedef struct llna_ss
{
    gsl_matrix * Ucov_ss;
    gsl_matrix * Vcov_ss;
    gsl_vector * Umu_ss;
    gsl_vector * Vmu_ss;
    gsl_matrix * Ubeta_ss;
    gsl_matrix * Vbeta_ss;

    double cov_ss;
    double Undata;
    double Vndata;
    double nratings;
} llna_ss;


/*
 * function declarations
 *
 */
void predict_y(gsl_matrix * Ucorpus_lambda,gsl_matrix * Vcorpus_lambda, ratings* r, gsl_vector * predict_r);
void get_rmse(ratings * r, gsl_vector * predict_r, double * rmse);
llna_model* read_llna_model(char*, int, int);
void write_llna_model(llna_model*, char*);
llna_model* new_llna_model(int, int, int, int);
llna_model* random_init(int, int, int, int);
llna_model* corpus_init(int, corpus*);
llna_ss * new_llna_ss(llna_model*);
void del_llna_ss(llna_ss*);
void reset_llna_ss(llna_ss*);
void write_ss(llna_ss*);

#endif
