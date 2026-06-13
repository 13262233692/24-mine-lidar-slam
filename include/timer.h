#pragma once

#include <chrono>
#include <string>
#include <iostream>
#include <iomanip>

namespace mine_slam {

class Timer {
public:
    explicit Timer(const std::string& name = "")
        : name_(name)
        , start_(std::chrono::high_resolution_clock::now()) {}

    void reset() {
        start_ = std::chrono::high_resolution_clock::now();
    }

    double elapsedSeconds() const {
        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = now - start_;
        return elapsed.count();
    }

    double elapsedMilliseconds() const {
        return elapsedSeconds() * 1000.0;
    }

    void print(const std::string& message = "") const {
        double ms = elapsedMilliseconds();
        if (!message.empty()) {
            std::cout << "[Timer] " << message << ": "
                      << std::fixed << std::setprecision(2) << ms << " ms" << std::endl;
        } else if (!name_.empty()) {
            std::cout << "[Timer] " << name_ << ": "
                      << std::fixed << std::setprecision(2) << ms << " ms" << std::endl;
        }
    }

private:
    std::string name_;
    std::chrono::high_resolution_clock::time_point start_;
};

class ScopedTimer {
public:
    explicit ScopedTimer(const std::string& name)
        : name_(name)
        , timer_(name) {}

    ~ScopedTimer() {
        timer_.print(name_ + " done");
    }

private:
    std::string name_;
    Timer timer_;
};

} // namespace mine_slam
