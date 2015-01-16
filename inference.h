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

#ifndef LLNA_INFERENCE_H
#define LLNA_INFERENCE_H

#define NEWTON_THRESH 1e-10

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <stdlib.h>
#include <stdio.h>

#include "corpus.h"
#include "ctm.h"
#include "gsl-wrappers.h"

//这是一个文档的变分参数，得到的结果存入整个文集的变分参数corpus_lambda等中
typedef struct llna_var_param {
    gsl_vector * Unu;
    gsl_vector * Vnu;

    gsl_vector * Ulambda;
    gsl_vector * Vlambda;

    double Uzeta;
    double Vzeta;

    gsl_matrix * Uphi;
    gsl_matrix * Vphi;

    gsl_matrix * Ulog_phi;
    gsl_matrix * Vlog_phi;

    int niter;
    short converged;
    double lhood;
    gsl_vector * Utopic_scores;
    gsl_vector * Vtopic_scores;
} llna_var_param;

typedef struct llna_corpus_var {
    gsl_matrix* Ucorpus_lambda;
    gsl_matrix* Vcorpus_lambda;
    gsl_matrix* Ucorpus_nu;
    gsl_matrix* Vcorpus_nu;
    gsl_matrix* Ucorpus_phi_sum;
    gsl_matrix* Vcorpus_phi_sum;
} llna_corpus_var;


typedef struct bundle {
    llna_var_param * var;
    llna_model * mod;
    doc * Udoc;
    doc * Vdoc;
    ratings * r;
    llna_corpus_var * c_var;
    gsl_vector * Usum_phi;
    gsl_vector * Vsum_phi;
} bundle;


/*
 * functions
 *
 */

void expected_theta(llna_var_param *var, doc* d, llna_model *mod, gsl_vector* v);

void free_llna_var_param(llna_var_param *);
void free_llna_Vvar_param(llna_var_param * v);
void free_llna_Uvar_param(llna_var_param * v);
double fixed_point_iter_i(int, llna_var_param *, llna_model *, doc *);

ratings*  get_uratings(ratings * r, int nusers);
ratings * get_vratings(ratings * r, int nitems);

void  init_corpus_var(llna_corpus_var * c_var, char* start);
void init_temp_vectors(int size);
void init_Uvar_unif(llna_var_param * var, doc * Udoc, llna_model * mod);
void init_Vvar_unif(llna_var_param * var, doc * Vdoc, llna_model * mod);
void init_Uvar(llna_var_param * var, doc * Udoc, llna_model * mod, gsl_vector *Ulambda, gsl_vector* Unu);
void init_Vvar(llna_var_param * var, doc * Vdoc, llna_model * mod, gsl_vector *Vlambda, gsl_vector* Vnu);
//void init_var(llna_var_param *var, doc *doc, llna_model *mod, gsl_vector *lambda, gsl_vector *nu);




llna_corpus_var * new_llna_corpus_var(int nusers, int nitems, int k);



llna_var_param* new_llna_Uvar_param(int, int);
llna_var_param* new_llna_Vvar_param(int, int);

int opt_Ulambda(llna_var_param * var, doc * Udoc, ratings * r_u, llna_model * mod, llna_corpus_var * c_var);
int opt_Vlambda(llna_var_param * var, doc * Vdoc, ratings * r_v, llna_model * mod, llna_corpus_var * c_var);
void opt_Uphi(llna_var_param * var, doc * Udoc, llna_model * mod);
void opt_Vphi(llna_var_param * var, doc * Vdoc, llna_model * mod);

void opt_Unu(llna_var_param * var, doc * Udoc,ratings * r_u, llna_model * mod, llna_corpus_var * c_var);
void opt_Vnu(llna_var_param * var, doc * Vdoc,ratings * r_v, llna_model * mod, llna_corpus_var * c_var);
void opt_Unu_i(int i, llna_var_param * var, ratings * r_ui, llna_model * mod, doc * d ,llna_corpus_var * c_var);
void opt_Vnu_i(int i, llna_var_param * var, ratings * r_vj, llna_model * mod, doc * d,llna_corpus_var * c_var);

int opt_Uzeta(llna_var_param * var, llna_model * mod);
int opt_Vzeta(llna_var_param * var, llna_model * mod);


void Ulhood_bnd(llna_var_param* var, doc* Udoc, ratings * r_ui, llna_model* mod, llna_corpus_var * c_var);
void Vlhood_bnd(llna_var_param* var, doc* Vdoc, ratings * r_ui, llna_model* mod, llna_corpus_var * c_var);
double Uvar_inference(llna_var_param * var, doc * Udoc, ratings* r, llna_model * mod, llna_corpus_var * c_var);
double Vvar_inference(llna_var_param * var, doc * Vdoc, ratings* r, llna_model * mod, llna_corpus_var * c_var);

void update_Vexpected_ss(llna_var_param* var, doc* Vdoc, ratings * r_vj,llna_corpus_var * c_var, llna_ss* ss);
void update_Uexpected_ss(llna_var_param* var, doc* Udoc, ratings * r_ui, llna_corpus_var * c_var,llna_ss* ss);
void update_expected_ss(llna_var_param* , doc*, doc*, ratings *, llna_ss*);



double sample_lhood(llna_var_param* var, doc* d, llna_model* mod);

double log_mult_prob(doc* d, gsl_vector* theta, gsl_matrix* log_beta);

void write_word_assignment(FILE* f, doc* d, gsl_matrix* phi);
void write_c_var(llna_corpus_var * c_var, char * root);

#endif
