/*
PLEASE WRITE DOWN NAME AND UID BELOW BEFORE SUBMISSION
* Filename: llama2_3035898913.c
* Student name and number: Davinne Valeria 3035898913
* Development platform: Ubuntu Docker container
* Remark: Complete all features
*/

#define _GNU_SOURCE // keep this line
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <sys/resource.h>
#include "utilities.h"

// YOUR CODE STARTS HERE

// ThreadData helps us to pass the arguments to worker threads
typedef struct {
    float* out; // output of multiplication
    float* vec; // input vector
    float* mat; // input matrix
    int col;    // number of columns
    int row;    // number of rows
} ThreadData;

// additional header file
#include <pthread.h>
#include <semaphore.h>

// global variables
struct rusage main_usage;        // get usage for main thread
struct rusage thread_usage[128]; // get usage for worker threads
int num_threads;                 // store number of threads globally
int working_threads = 0;         // helper variable to check if all threads have finished their jobs
int thread_ids[128];             // pass the information of thread id to worker threads
pthread_t threads[128];          // worker threads
ThreadData thread_data;          // struct to pass mat_vec_mul arguments to worker threads

sem_t sem_thread[128];           // semaphore for worker threads to help them sleep and wake up
sem_t thread_lock;               // semaphore to protect operations on a global variable, in this case working_threads
sem_t work_done;                 // semaphore to notify main thread that worker threads have finished

void *thr_func(void *arg);

int create_mat_vec_mul(int thr_count) {
    num_threads = thr_count; // pass number of threads to the global variable

    // initiate semaphore 
    sem_init(&thread_lock, 0, 1); 
    sem_init(&work_done, 0, 0);

    for (int i = 0; i < thr_count; i++) {
        thread_ids[i] = i; // assign thread ids
        sem_init(&sem_thread[i], 0, 0); // initiate threads' semaphore
    }

    for (int i = 0; i < thr_count; i++) {
        pthread_create(&threads[i], NULL, &thr_func, &thread_ids[i]); // create threads, passing the thread id as the argument
    }
}


void mat_vec_mul(float* out, float* vec, float* mat, int col, int row) {
    // assign values to the thread_data based on the arguments of mat_vec_mul
    for (int i = 0; i < num_threads; i++){
        thread_data.out = out;
        thread_data.vec = vec;
        thread_data.mat = mat;
        thread_data.col = col;
        thread_data.row = row;
    }
    
    working_threads = 0; // initiate value

    // wake up worker threads one by one
    for (int i = 0; i < num_threads; i++){
        sem_post(&sem_thread[i]);
    }

    // wait until all worker threads finish their jobs
    sem_wait(&work_done);
}


int destroy_mat_vec_mul() {

    for (int i = 0; i < num_threads; i++){
        // assign 0 the row to tell threads to terminate
        thread_data.row = 0;
        // wake up worker thread to terminate
        sem_post(&sem_thread[i]);
        // wait until worker thread terminate
        pthread_join(threads[i], NULL);
    }

    // destroy all worker threads semaphore
    for (int i = 0; i < num_threads; i++){
        sem_destroy(&sem_thread[i]);
    }

    // destroy the rest of the semaphores
    sem_destroy(&thread_lock);
    sem_destroy(&work_done);

    // collect usage data of main thread
    getrusage(RUSAGE_SELF, &main_usage);
    printf("main thread - user: %ld.%06ld s, system: %ld.%06ld s\n", main_usage.ru_utime.tv_sec, main_usage.ru_utime.tv_usec, main_usage.ru_stime.tv_sec, main_usage.ru_stime.tv_usec);
    return 0;
}


void *thr_func(void *arg) {
    while (1) {
        int thread_id = *(int*) arg; // collect data of thread id
        sem_wait(&sem_thread[thread_id]); // sleep until main thread wakes the worker thread

        int start_row, end_row; // variables to help worker threads know where to start and stop the matrix multiplication
        // retrieve data from thread_data
        float* out = thread_data.out;
        float* vec = thread_data.vec;
        float* mat = thread_data.mat;
        int col = thread_data.col;
        int row = thread_data.row;
        
        int row_nums = row / num_threads; // helper variable

        // Check if row = 0, denotes that the main thread tells the worker threads to terminate
        if (row == 0) { 
            getrusage(RUSAGE_THREAD, &thread_usage[thread_id]); // collect usage data of the worker thread
            printf("Thread %d has completed - user: %ld.%06ld s, system: %ld.%06ld s\n", thread_id, thread_usage[thread_id].ru_utime.tv_sec, thread_usage[thread_id].ru_utime.tv_usec, thread_usage[thread_id].ru_stime.tv_sec, thread_usage[thread_id].ru_stime.tv_usec);
            pthread_exit(NULL); // terminate the thread
        }

        // The part below helps us to calculate the starting and ending row of the current worker thread to calculate
        if (row % num_threads == 0){ // if the number of rows divisible by number of threads
            start_row = thread_id * row_nums;
            end_row = (thread_id+1) * row_nums;
        } else {  // if the number of rows is not divisible by number of threads
            if (thread_id != num_threads - 1){ // if the worker thread is not the last one (with largest thread id)
                start_row = thread_id * (ceil(row_nums));
                end_row = (thread_id+1) * (ceil(row_nums));
            }
            else if ((num_threads-1) * (ceil(row_nums)) < row){ // if this is the last thread and there are rows left to calculate
                start_row = (num_threads-1) * (ceil(row_nums));
                end_row = row;
            }
            else { // if the last worker thread does not have any rows left to calculate
                start_row = row-1;
                end_row = row-1;
            }
        }

        // this for loop is where the matrix multiplication happens
        for (int i = start_row; i < end_row; i++){
            float val = 0.0f;
            for (int j = 0; j < col; j++){
                val += mat[i * col + j] * vec[j];
            }
            out[i] = val; // assign result per row to "out"
        }

        // use lock to modify global variable (working_threads)
        sem_wait(&thread_lock);
        working_threads += 1;    
        // check if working_threads is equal to number of threads, meaning all worker threads have finished their jobs
        // if yes, inform the main thread
        if (working_threads == num_threads) sem_post(&work_done);  
        sem_post(&thread_lock); // release lock
    }
}

// YOUR CODE ENDS HERE

int transformer(int token, int pos, LLMConfig* p, LLMRuntime* s, LLMWeight* w) {
    
    // a few convenience variables
    int dim = p->dim, hidden_dim =  p->hidden_dim, head_size = p->dim / p->n_heads;

    // copy the token embedding into x
    memcpy(s->x, &(w->token_embedding_table[token * dim]), dim*sizeof(float));

    // forward all the layers
    for(int l = 0; l < p->n_layers; l++) {

        // Attention
        {
            // attention normalization
            normalize(s->xb, s->x, w->rms_att_weight + l*dim, dim);

            // q, k, v = w_q @ x, w_k @ x, w_v @ x, respectively
            mat_vec_mul(s->q, s->xb, w->wq + l*dim*dim, dim, dim);
            mat_vec_mul(s->k, s->xb, w->wk + l*dim*dim, dim, dim);
            mat_vec_mul(s->v, s->xb, w->wv + l*dim*dim, dim, dim);

            // apply positional embedding
            position_embedding(s->q, s->k, w, pos, p->dim, p->n_heads);

            // save intermediate result for later reference
            key_value_cache(l, pos, p, s);
            
            // attention calculation
            attention(l, pos, p, s, w);

            // wo @ x to get final result
            mat_vec_mul(s->xb2, s->xb, w->wo + l*dim*dim, dim, dim);

            // residual connection back into x
            accum(s->x, s->xb2, dim);
        }
    
        // Feed-Forward Network: w2 @ (silu(w1 @ x) * (w3 @ x)), * is element-wise multiply
        {
            // FFN Normalization
            normalize(s->xb, s->x, w->rms_ffn_weight + l*dim, dim);

            // w1 @ x
            mat_vec_mul(s->h1, s->xb, w->w1 + l*dim*hidden_dim, dim, hidden_dim);
            // silu(w1 @ x)
            silu(s->h1, hidden_dim);
            // w3 @ x
            mat_vec_mul(s->h2, s->xb, w->w3 + l*dim*hidden_dim, dim, hidden_dim);
            // silu(w1 @ x) * (w3 @ x)
            element_wise_mul(s->h1, s->h2, hidden_dim);
            // w2 @ (silu(w1 @ x) * (w3 @ x))
            mat_vec_mul(s->xb, s->h1, w->w2 + l*dim*hidden_dim, hidden_dim, dim);

            // residual connection
            accum(s->x, s->xb, dim);
        }
    }
    
    // final normalization
    normalize(s->x, s->x, w->rms_final_weight, dim);
    // classifier into logits
    mat_vec_mul(s->logits, s->x, w->token_embedding_table, p->dim, p->vocab_size);
    // apply the temperature to the logits
    for (int q=0; q<p->vocab_size; q++) { s->logits[q] /= 0.9f; }
    // apply softmax to the logits to get the probabilities for next token
    softmax(s->logits, p->vocab_size);
    // now sample from this distribution to get the next token
    return sample(s->logits, p->vocab_size);
}

int main(int argc, char* argv[]) {

    unsigned int seed;
    int thr_count;

    if (argc == 3) {
        seed = atoi(argv[1]);
        thr_count = atoi(argv[2]);
    } else {
        printf("Usage: ./compiled <seed> <thr_count>\n");
        return 1;
    }

    // Initialize
    srand(seed);
    create_mat_vec_mul(thr_count);

    // load model
    LLMConfig config;
    LLMWeight weights;
    if (load_LLM_Config_Weight(&config, &weights) == 1) { return 1; }

    // load tokenizer
    char** vocab = malloc(config.vocab_size * sizeof(char*));
    if (load_tokenizer(vocab, config.vocab_size) == 1) { return 1; }

    // create and init the application LLMRuntime
    LLMRuntime state;
    malloc_LLMRuntime(&state, &config);
    
    // the current position we are in
    long start = time_in_ms();

    int next, token = 1, pos = 0; // token = 1 -> <START>
    while (pos < config.seq_len) {

        // forward the transformer to get logits for the next token
        next = transformer(token, pos, &config, &state, &weights);

        printf("%s", vocab[next]);
        fflush(stdout); // force print

        token = next;
        pos++;
    }

    long end = time_in_ms();
    printf("\n\nlength: %d, time: %f s, achieved tok/s: %f\n", config.seq_len, (double)(end-start)/1000, config.seq_len / (double)(end-start)*1000);

    // cleanup
    destroy_mat_vec_mul();
    free_LLMRuntime(&state);
    free_LLMWeight(&weights);
    for (int i = 0; i < config.vocab_size; i++) { free(vocab[i]); }
    free(vocab);
    return 0;
}
