#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <vector>
#include <filesystem>


namespace fs = std::filesystem;

class ConfigParser {
public:
    std::pair<int, int> get_n_particles() {
        std::string filePath = "params/nexternal.inc";
        std::ifstream file(filePath);

        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << filePath << std::endl;
            throw std::runtime_error("Failed to open file: " + filePath);
        }

        std::string line;
        std::getline(file, line);
        std::getline(file, line);

        auto pos = line.find("=") + 1;
        auto end_pos = line.find(")", pos);
        int n_external = std::stoi(line.substr(pos, end_pos - pos));

        std::getline(file, line);
        std::getline(file, line);

        pos = line.find("=") + 1;
        end_pos = line.find(")", pos);

        int n_incoming = std::stoi(line.substr(pos, end_pos - pos));

        file.close();

        return std::make_pair(n_external, n_incoming);
    }

    std::vector<std::string> get_tags() {
        std::vector<std::string> masses;

        std::string filePath = "params/pmass.inc";
        std::ifstream file(filePath);

        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << filePath << std::endl;
            throw std::runtime_error("Failed to open file: " + filePath);
        }

        std::string line;
        while (getline(file, line)) {
            std::string mass = line.substr(line.find("=") + 1);

            mass.erase(mass.find_last_not_of(" \n\r\t") + 1);
            mass = mass.substr(mass.rfind("(") + 1, (mass.rfind(")") - mass.rfind("(") - 1));

            masses.push_back(mass);
        }

        file.close();
        return masses;
    }

    std::pair<std::vector<double>, std::vector<double>> _parse_masses(int nIncoming, const std::vector<std::string>& massTags) {
        std::string filePath = "params/param.log";
        std::ifstream file(filePath);

        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << filePath << std::endl;
            throw std::runtime_error("Failed to open file: " + filePath);
        }

        std::string line;
        double Gf, aEW, MZ;

        while (getline(file, line)) {
            std::istringstream iss(line);
            std::vector<std::string> tokens{std::istream_iterator<std::string>{iss}, std::istream_iterator<std::string>{}};

            if (tokens[1] == "mdl_gf") Gf = std::stod(tokens[7]);
            else if (tokens[1] == "aewm1") aEW = 1.0 / std::stod(tokens[7]);
            else if (tokens[1] == "mdl_mz") MZ = std::stod(tokens[7]);
        }

        file.close();

        std::vector<double> masses(massTags.size(), 0);

        for (size_t j = 0; j < massTags.size(); ++j) {
            if (massTags[j] == "ZERO")
                masses[j] = 0.0;
            else if (massTags[j] == "MDL_MW")
                masses[j] = std::sqrt(MZ*MZ/2.0 + std::sqrt(MZ*MZ*MZ*MZ/4.0 - (aEW*M_PI*MZ*MZ)/(Gf*std::sqrt(2.0))));
            else {
                file.open(filePath);
                while (getline(file, line)) {
                    if (line.find(" " + massTags[j].substr(5) + " ") != std::string::npos) {
                        std::istringstream iss(line);
                        std::vector<std::string> tokens{std::istream_iterator<std::string>{iss}, std::istream_iterator<std::string>{}};
                        masses[j] = std::stod(tokens[7]);
                        break;
                    }
                }
                file.close();
            }
        }

        std::vector<double> incoming(masses.begin(), masses.begin() + nIncoming);
        std::vector<double> outgoing(masses.begin() + nIncoming, masses.end());

        return {incoming, outgoing};
    }

    std::pair<std::vector<double>, std::vector<double>> parse_masses() {
        auto n_particles = get_n_particles();
        auto tags = get_tags();

        return _parse_masses(n_particles.second, tags);
    }
};
