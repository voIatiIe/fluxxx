#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <stdexcept>


class Config {
public:
    Config(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open config file");
        }

        std::string line;
        while (std::getline(file, line)) {
            auto delimiterPos = line.find('=');
            if (delimiterPos != std::string::npos) {
                std::string key = line.substr(0, delimiterPos);
                std::string value = line.substr(delimiterPos + 1);
                settings[key] = value;
            }
        }

        file.close();
    }

    template <typename T>
    T get(const std::string& key) {
        if (settings.find(key) == settings.end()) {
            throw std::runtime_error("Key not found in configuration");
        }

        std::istringstream iss(settings[key]);
        T value;
        if (!(iss >> value)) {
            throw std::runtime_error("Invalid type conversion for key: " + key);
        }

        return value;
    }

private:
    std::unordered_map<std::string, std::string> settings;
};

template <>
std::string Config::get<std::string>(const std::string& key) {
    if (settings.find(key) == settings.end()) {
        throw std::runtime_error("Key not found in configuration");
    }
    return settings[key];
}
