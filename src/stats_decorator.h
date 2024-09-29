#ifndef STATS_DECORATOR_H
#define STATS_DECORATOR_H

#include <libutils/timer.h>
#include <utility>

class StatsGenerator {
public:
    StatsGenerator(unsigned int iters)
    : iters_(iters)
    {
    }

    template<typename Action, typename ...Args>
    void runBenchmark(unsigned n, Action& action, const Args& ...args) {
        timer t;

        for (unsigned i = 0; i < iters_; ++i) {
            action(args...);
            t.nextLap();
        }

        std::cout << "GPU:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU:     " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;

    }

private:
    unsigned int iters_;
};



#endif // STATS_DECORATOR_H
