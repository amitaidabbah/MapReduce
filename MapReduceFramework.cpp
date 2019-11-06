#include "MapReduceFramework.h"
#include <pthread.h>
#include <cstdio>
#include <atomic>
#include <unordered_map>
#include <iostream>
#include <algorithm>
#include "Barrier.h"
#include <utility>
#include <semaphore.h>
#include <iostream>

struct Job;

struct ThreadContext
{
    pthread_t *thread;
    Job *job;
    IntermediateVec intervec;
};


struct Job
{
    const MapReduceClient *client;


    std::vector<ThreadContext *> contexts;
    std::vector<pthread_t> threads;
    const InputVec *inputVec;
    std::vector<IntermediateVec> to_reduce;
    OutputVec *outputVec;

    int num_threads;
    pthread_t *shuffle_thread = nullptr;

    JobState jobState = {UNDEFINED_STAGE, 0};

    Barrier barrier;
    std::atomic_int counter{0};
    std::atomic_ulong proccesed{0};
    std::atomic_ulong stage_inputs{0};
    pthread_mutex_t input_vec_mutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_t output_vec_mutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_t reduce_mutex = PTHREAD_MUTEX_INITIALIZER;
    sem_t sem;
    std::atomic_bool shuffle_is_done{false};
    bool wait_bool = false;

    Job(int num_threads, const InputVec *inv, OutputVec *outv, const MapReduceClient *client) :
            client(client),
            inputVec(inv),
            outputVec(outv),
            num_threads(
                    num_threads),
            barrier(num_threads)
    {
        if (sem_init(&sem, 0, 0) != 0)
        {
            std::cout << "semaphore init error\n";
        }
    }

    ~Job()
    {
    }
};

//-------Function Declarations------------------------------------------------------------------------------------------
void *start_job(void *);

// ------Variables------------------------------------------------------------------------------------------------------
static std::vector<Job *> jobs;

//-------Main Code------------------------------------------------------------------------------------------------------
JobHandle startMapReduceJob(const MapReduceClient &client,
                            const InputVec &inputVec, OutputVec &outputVec,
                            int multiThreadLevel)
{
    Job *main_job = new Job(multiThreadLevel, &inputVec, &outputVec, &client);
    jobs.push_back(main_job);
    for (int i = 0; i < multiThreadLevel; ++i)
    {
        auto t = new (pthread_t);
        auto tc = new ThreadContext{nullptr, main_job};
        tc->thread = t;
        main_job->contexts.insert(main_job->contexts.begin(), tc);
        if (i == 0)
        {
            main_job->shuffle_thread = t;
        }
        if (pthread_create(t, nullptr, start_job, tc) != 0)
        {
            std::cout << "error creating thread: " << i << ", exiting\n";
            exit(1);
        }
    }
    return (JobHandle) main_job;

}

void mutex_lock(pthread_mutex_t *mutex)
{
    if (pthread_mutex_lock(mutex) != 0)
    {
        std::cout << "error on pthread_mutex_lock\n";
        exit(1);
    }
}

void mutex_unlock(pthread_mutex_t *mutex)
{
    if (pthread_mutex_unlock(mutex) != 0)
    {
        std::cout << "error on pthread_mutex_unlock\n";
        exit(1);
    }
}

void do_map_sort(void *thread_context)
{
    auto tc = (ThreadContext *) thread_context;
    Job *job = tc->job;
    job->jobState.stage = MAP_STAGE;
    K1 *current_key;
    V1 *current_val;
    bool done = false;
    mutex_lock(&job->input_vec_mutex);
    long unsigned int old_value = (job->counter)++;
    if (old_value < job->inputVec->size())
    {
        current_key = job->inputVec->at(old_value).first;
        current_val = job->inputVec->at(old_value).second;
    } else
    {
        done = true;
    }

    mutex_unlock(&job->input_vec_mutex);
    while (!done)
    {
        job->client->map(current_key, current_val, tc);
        job->proccesed++;
        job->jobState.stage = MAP_STAGE;
        mutex_lock(&job->input_vec_mutex);
        long unsigned int old_value = job->counter++;
        if (old_value < job->inputVec->size())
        {
            current_key = job->inputVec->at(old_value).first;
            current_val = job->inputVec->at(old_value).second;
        } else
        {
            done = true;
        }
        mutex_unlock(&job->input_vec_mutex);

    }

    //perform sorting on the intermediate array:
    std::sort(tc->intervec.begin(), tc->intervec.end(),
              [](const IntermediatePair &p1, const IntermediatePair &p2) -> bool
              { return *(p1.first) < *(p2.first); });
}

void do_shuffle(void *thread_context)
{
    auto tc = (ThreadContext *) thread_context;
    Job *job = tc->job;

    job->jobState.stage = REDUCE_STAGE;
    job->stage_inputs = 0;
    job->proccesed = 0;

    int counter = 0;
    for (auto &context : job->contexts)
    {
        job->stage_inputs += context->intervec.size();
        counter += context->intervec.size();
    }
    while (counter > 0)
    {
        K2 *maxK = nullptr;
        for (auto &context : job->contexts)
        {
            if (!context->intervec.empty())
            {
                IntermediatePair pair = context->intervec.back();
                if (maxK == nullptr || *maxK < *pair.first)
                {
                    maxK = pair.first;
                }
            }

        }
        IntermediateVec intermediateVec;
        for (auto &context : job->contexts)
        {
            if (context->intervec.empty())
            {
                continue;
            }

            while (!(context->intervec.empty()) && !(*maxK < *context->intervec.back().first) &&
                   !(*context->intervec.back().first < *maxK))
            {
                IntermediatePair &intermediatePair = context->intervec.back();
                intermediateVec.push_back(intermediatePair);
                context->intervec.pop_back();
                counter--;
            }
        }
        mutex_lock(&job->reduce_mutex);
        job->to_reduce.push_back(intermediateVec);
        mutex_unlock(&job->reduce_mutex);
        sem_post(&job->sem);
    }

    job->shuffle_is_done = true;
}

void do_reduce(void *thread_context)
{
    auto tc = (ThreadContext *) thread_context;
    Job *job = tc->job;

    while (!(job->shuffle_is_done) || !(job->to_reduce.empty()))
    {
        sem_wait(&job->sem);
        // Cond
        if (!(job->shuffle_is_done) || !(job->to_reduce.empty()))
        {
            mutex_lock(&job->reduce_mutex);
            IntermediateVec cur_seq = job->to_reduce.back();
            job->to_reduce.pop_back();
            mutex_unlock(&job->reduce_mutex);
            job->client->reduce(&cur_seq, job);
            job->proccesed += cur_seq.size();
        }


    }
    sem_post(&job->sem);
//    pthread_exit(nullptr);
}

void *start_job(void *thread_context)
{
    auto tc = (ThreadContext *) thread_context;
    Job *job = tc->job;
    // do map/sort
    do_map_sort(tc);
    job->barrier.barrier();
    //do shuffle
    if (pthread_self() == *job->shuffle_thread)
    {
        do_shuffle(tc);
    }
    //do reduce
    do_reduce(tc);
    return nullptr;
}


void waitForJob(JobHandle main_job)
{
    auto *jc = (Job *) main_job;
    if (!jc->wait_bool)
    {
        jc->wait_bool = true;
        for (auto &context : jc->contexts)
        {

            if (*context->thread == (pthread_t) NULL)
            {
                continue;
            }
            if (pthread_join(*context->thread, nullptr))
            {
                std::cerr << "Error using pthread_join.\n";
                exit(1);
            }
        }
    }


}

void getJobState(JobHandle main_job, JobState *state)
{
    Job *job = (Job *) main_job;
    job->jobState.percentage = (float) job->proccesed * 100 / job->stage_inputs;
    *state = job->jobState;
}

void closeJobHandle(JobHandle main_job)
{
    Job *job = (Job *) main_job;
    waitForJob(job);

    for (ThreadContext *context: job->contexts)
    {
        delete context->thread;
        delete context;
    }
    delete job;

}

void emit2(K2 *key, V2 *value, void *context)
{
    auto *tc = (ThreadContext *) context;
    IntermediatePair p(key, value);
    tc->intervec.push_back(p);

}

void emit3(K3 *key, V3 *value, void *main_job)
{
    Job *job = (Job *) main_job;
    OutputPair p(key, value);
    mutex_lock(&job->output_vec_mutex);
    job->outputVec->push_back(p);
    mutex_unlock(&job->output_vec_mutex);
}
