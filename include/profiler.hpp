#include <iostream>
#include <chrono>

using duration = std::chrono::microseconds;

struct profiler
{
    std::string name;
    std::chrono::high_resolution_clock::time_point p;

    profiler(std::string const &name) : name(name), p(std::chrono::high_resolution_clock::now()) {}
    ~profiler()
    {
        auto d = std::chrono::high_resolution_clock::now() - p;
        std::cout << name << ": "
                  << std::chrono::duration_cast<duration>(d).count()
                  << "[µs]" << std::endl;
    }
};

#define PROFILE(pbn) profiler _pfinstance(pbn)
