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

#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_rng.h>
#include <time.h>
#include <assert.h>

#include "corpus.h"

/*
 * read corpus from a file
 *
 */

corpus* read_data(const char* data_filename)
{
    FILE *fileptr;
    int length, count, word, n, nd, nw, corpus_total = 0;
    corpus* c;

    printf("reading data from %s\n", data_filename);
    c = malloc(sizeof(corpus));
    fileptr = fopen(data_filename, "r");
    nd = 0; nw = 0;                    //初始化，0个doc，0个word
    c->docs = malloc(sizeof(doc) * 1);  //这里之分配了一个doc的内存
    while ((fscanf(fileptr, "%10d", &length) != EOF))
    {
    //printf("%d\n",sizeof(doc));  //这里申请了4个int的空间，值为16,实际是两个int，两个int指针头。32位系统，一个int32位，4个字节
	c->docs = (doc*) realloc(c->docs, sizeof(doc)*(nd+1));
	//c->docs = (doc*) realloc(c->docs, 16*(nd+1));
	//printf("%d\n",sizeof(c->docs));  //这是什么情况，重新分配的大小到哪儿？？
	//printf("%d\n",sizeof(c));

	c->docs[nd].nterms = length;
	c->docs[nd].total = 0;
	c->docs[nd].word = malloc(sizeof(int)*length);
	c->docs[nd].count = malloc(sizeof(int)*length);
	for (n = 0; n < length; n++)
	{
	    fscanf(fileptr, "%10d:%10d", &word, &count);
	    word = word - OFFSET;
	    //c->docs[nd].word[n] = word;

	    c->docs[nd].word[n] = word;
	    c->docs[nd].count[n] = count;
	    c->docs[nd].total += count;
	    if (word >= nw) { nw = word + 1; }
	}
	corpus_total += c->docs[nd].total;
        nd++;
    }
    fclose(fileptr);
    c->ndocs = nd;
    c->nterms = nw;
    printf("number of docs    : %d\n", nd);
    printf("number of terms   : %d\n", nw);
    printf("total             : %d\n", corpus_total);
    return(c);
}

/*
 * read ratings from a file
 *
 */

ratings* read_rating(const char* data_filename) {


    FILE *fileptr;
    int nr, user, item;
    double rating;
    ratings* r;

    printf("reading ratings from %s\n", data_filename);
    fileptr = fopen(data_filename, "r");
    nr = 0;
    r = malloc(sizeof(ratings));

    //必须要对结构提内部的指针先初始化，才能利用realloc重新分配大小
	r->users = malloc(sizeof(int)*1);
	r->items = malloc(sizeof(int)*1);
	r->ratings = malloc(sizeof(double)*1);


	//int a;
    while (!feof(fileptr))
    {
		r->users = (int*) realloc(r->users, sizeof(int)*(nr+1));
		r->items = (int*) realloc(r->items, sizeof(int)*(nr+1));
		r->ratings = (double*) realloc(r->ratings, sizeof(double)*(nr+1));

		if((fscanf(fileptr, "%d %d %lf", &user, &item,&rating)!=EOF)) {
			//printf("%10d,%10d\n", user, item);
			r->users[nr] = user;
			r->items[nr] = item;
			r->ratings[nr] = rating;
/*			if(rating>10)
				r->ratings[nr] = 5;
			else
				r->ratings[nr] = 4/9.0*(rating-1)+1;*/
			//r->ratings[nr] = 1.0;
			//printf("r->users:%d,%d,%f\n", r->users[nr], r->items[nr], r->ratings[nr]);
			nr++;
			//printf("%d\n",nr);
		}
    }

    fclose(fileptr);
    r->nratings = nr;
    printf("number of ratings : %d\n", nr);
	return(r);

}


/*
 * print document
 *
 */

void print_doc(doc* d)
{
    int i;
    printf("total : %d\n", d->total);
    printf("nterm : %d\n", d->nterms);
    for (i = 0; i < d->nterms; i++)
    {
        printf("%d:%d ", d->word[i], d->count[i]);
    }
}


/*
 * write a corpus to file
 *
 */

void write_corpus(corpus* c, char* filename)
{
    int i, j;
    FILE * fileptr;
    doc * d;

    fileptr = fopen(filename, "w");
    for (i = 0; i < c->ndocs; i++)
    {
        d = &(c->docs[i]);
        fprintf(fileptr, "%d", d->nterms);
        for (j = 0; j < d->nterms; j++)
        {
            fprintf(fileptr, " %d:%d", d->word[j], d->count[j]);
        }
        fprintf(fileptr, "\n");
    }
}


void init_doc(doc* d, int max_nterms)
{
    int i;
    d->nterms = 0;
    d->total = 0;
    d->word = malloc(sizeof(int) * max_nterms);
    d->count = malloc(sizeof(int) * max_nterms);
    for (i = 0; i < max_nterms; i++)
    {
        d->word[i] = 0;
        d->count[i] = 0;
    }
}


/*
 * return the 'n'th word of a document
 * (note order has been lost in the representation)
 *
 */

int remove_word(int n, doc* d)
{
    int i = -1, word, pos = 0;
    do
    {
        i++;
        pos += d->count[i];
        word = d->word[i];
    }
    while (pos <= n);
    d->total--;
    d->count[i]--;
    assert(d->count[i] >= 0);
    return(word);
}


/*
 * randomly move some proportion of words from one document to another
 *
 */

void split(doc* orig, doc* dest, double prop)
{
    int w, i, nwords;

    gsl_rng * r = gsl_rng_alloc(gsl_rng_taus);
    time_t seed;
    time(&seed);
    gsl_rng_set(r, (long) seed);

    nwords = floor((double) prop * orig->total);
    if (nwords == 0) nwords = 1;
    init_doc(dest, nwords);
    for (i = 0; i < nwords; i++)
    {
        w = remove_word(floor(gsl_rng_uniform(r)*orig->total), orig);
        dest->total++;
        dest->nterms++;
        dest->word[i] = w;
        dest->count[i] = 1;
    }
}
