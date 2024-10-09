#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>

class timer {
protected:

    using timer_type = std::chrono::high_resolution_clock::time_point;

    double counter_;
    timer_type start_;
    int is_running_;
    
    std::vector<double> laps_;

public:
    explicit timer(bool paused = false)
    {
        counter_ = 0;
        is_running_ = 0;
        if (!paused)
            start();
    }

    void start()
    {
        if (is_running_) return;

        start_ = measure();
        is_running_ = 1;
    }

    void stop()
    {
        if (!is_running_) return;

        counter_ += diff(start_, measure());
        is_running_ = 0;
    }

    double nextLap()
    {
        double lap_time = elapsed();
        laps_.push_back(lap_time);
        restart();
        return lap_time;
    }

    void reset()
    {
        counter_ = 0;
        is_running_ = 0;
    }

    void restart()
    {
        reset();
        start();
    }

    double elapsed() const
    {
        double tm = counter_;

        if (is_running_)
            tm += diff(start_, measure());

        if (tm < 0)
            tm = 0;

        return tm;
    }
    
    const std::vector<double>& laps() const
    {
        return laps_;
    }

    // Note that this is not true averaging, if there is at least 5 laps - averaging made from 20% percentile to 80% percentile (See lapsFiltered)
    double lapAvg() const
    {
        std::vector<double> laps = lapsFiltered();
        
        double sum = 0.0;
        for (int i = 0; i < laps.size(); ++i) {
            sum += laps[i];
        }
        if (laps.size() > 0) {
            sum /= laps.size();
        }
        return sum;
    }

    // Note that this is not true averaging, if there is at least 5 laps - averaging made from 20% percentile to 80% percentile (See lapsFiltered)
    double lapStd() const
    {
        double avg = lapAvg();

        std::vector<double> laps = lapsFiltered();

        double sum2 = 0.0;
        for (int i = 0; i < laps.size(); ++i) {
            sum2 += laps[i] * laps[i];
        }
        if (laps.size() > 0) {
            sum2 /= laps.size();
        }
        return sqrt(std::max(0.0, sum2 - avg * avg));
    }

protected:

    std::vector<double> lapsFiltered() const
    {
        std::vector<double> laps = laps_;
        std::sort(laps.begin(), laps.end());

        unsigned int nlaps = laps.size();
        if (nlaps >= 5) {
            // Removing last 20% of measures
            laps.erase(laps.end() - nlaps/5, laps.end());
            // Removing first 20% of measures
            laps.erase(laps.begin(), laps.begin() + nlaps/5);
        }
        return laps;
    }

    static timer_type measure()
    {
        return std::chrono::high_resolution_clock::now();
    }

    static double diff(const timer_type &start, const timer_type &end)
    {
        long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        return double(microseconds) / 1000000;
    }
};
